#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes per-user throughput via Sionna ray tracing for mmWave and sub-6 GHz bands.
- PathSolver (ray tracing) runs on CPU (llvm_ad_mono_polarized).
- TensorFlow and throughput computations run on GPU.
- All TX and RX positions are transformed from the SUMO coordinate system to Sionna.
- Generates validation images with TX (RU) and RX positions.

Usage:
    source ~/sionna_env/bin/activate
    python compute_throughput.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path as PathlibPath

# --- Mitsuba: set variant ONCE per process ---
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

# ------------------ EXECUTION SETTINGS ------------------
USERS_CSV = "users_predicted_location_best60.csv"
FINAL_OUTPUT = "throughput_users.csv"
NUM_BLOCKS = 3           # number of segments to split the dataset into

IMG_EVERY = 200

def combine_blocks():
    """Reads all partial CSVs throughput_users_part_*.csv and generates the final output file."""
    print("[INFO] Combining partial results...")
    part_files = sorted(PathlibPath(".").glob("throughput_users_part_*.csv"))
    if not part_files:
        print("[WARN] No partial files found to combine.")
        return
    dfs = [pd.read_csv(f) for f in part_files]
    pd.concat(dfs, ignore_index=True).to_csv(FINAL_OUTPUT, index=False)
    print(f"[INFO] Combined output saved to: {FINAL_OUTPUT}")

def run_blocks():
    """Runs all blocks sequentially and concatenates partial results into the final CSV."""
    df = pd.read_csv(USERS_CSV)
    n_total = len(df)
    block_size = int(np.ceil(n_total / NUM_BLOCKS))
    print(f"[INFO] Dataset has {n_total} datapoints. Splitting into {NUM_BLOCKS} blocks (~{block_size} each).")

    for i in range(NUM_BLOCKS):
        start = i * block_size
        end = min((i + 1) * block_size, n_total)
        print(f"\n[INFO] Starting block {i+1}/{NUM_BLOCKS}: rows {start} to {end}...")
        run_segment(start, end)
        print(f"[INFO] Block {i+1}/{NUM_BLOCKS} done.\n")

    combine_blocks()

# ------------------ SEGMENT EXECUTION ------------------
def run_segment(start, end):
    print("[INFO] PathSolver configured for CPU (llvm_ad_mono_polarized)")

    import sionna.rt.radio_materials.itu
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver, Camera
    import tensorflow as tf
    from utils import (
        compute_steering_vector,
        apply_precoder,
        calc_thr_per_antenna,
        transform_xy_sumo_to_sionna,
    )

    # ------------------ Configs ------------------
    scenario_file = "./blender_scenario/ufg.xml"
    user_height = 1.5
    seed = 42
    tx_display_radius = 5
    rx_display_radius = 5
    MAP_ROT_DEG = 0
    OFFSET_XY = (-188.0, -10.0)
    out_csv = f"throughput_users_part_{start}_{end}.csv"

    xmin = 156.336460
    xmax = 1265.827054
    ymin = 0.230
    ymax = 836.226308
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    df_full = pd.read_csv(USERS_CSV)
    n_total = len(df_full)
    if end > n_total:
        end = n_total
    df = df_full.iloc[start:end].reset_index(drop=True)
    print(f"[INFO] Segment loaded: {start}–{end} ({len(df)} datapoints).")

    required_cols = [
        "id_user",
        "x_real", "y_real",
        "x_central_offline", "y_central_offline",
        "x_central_online", "y_central_online",
        "x_federated", "y_federated",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # ------------------ Adjust RX positions ------------------
    for prefix in ["real", "central_offline", "central_online", "federated"]:
        x_col, y_col = f"x_{prefix}", f"y_{prefix}"
        xr, yr = transform_xy_sumo_to_sionna(
            df[x_col].values, df[y_col].values,
            cx=cx, cy=cy, map_rot_deg=MAP_ROT_DEG, offset_xy=OFFSET_XY
        )
        df[x_col] = xr
        df[y_col] = yr
    print("[INFO] User coordinates transformed to the Sionna coordinate system")

    # ------------------ TX positions ------
    tx_position_without_sectors = np.array([
        [1098.536159, 159.954115, 15.0],
        [815.293013, 252.609317, 15.0],
        [1144.899793, 476.649208, 15.0],
        [1043.099012, 561.97574, 15.0],
        [817.878341, 683.808948, 15.0],
        [661.200095, 484.219558, 15.0],
        [307.593191, 339.676499, 15.0],
        [450.154461, 694.131334, 15.0],
        [492.699616, 212.960978, 15.0],
    ])

    tx_xr, tx_yr = transform_xy_sumo_to_sionna(
        tx_position_without_sectors[:, 0],
        tx_position_without_sectors[:, 1],
        cx=cx, cy=cy, map_rot_deg=MAP_ROT_DEG, offset_xy=OFFSET_XY,
    )
    tx_position_without_sectors[:, 0] = tx_xr
    tx_position_without_sectors[:, 1] = tx_yr

    # ------------------ Sectors (mmWave) ------------------
    origin_positions = np.array([
        [0, 0, 0],
        [0.5, 0, 0],
        [0.25, (0.5 * np.sqrt(3)) / 2, 0],
    ])
    tx_positions_with_sectors = np.array(
        [origin_positions + tx_pos for tx_pos in tx_position_without_sectors]
    ).reshape(-1, 3)
    tx_orientation = np.tile([
        [0, np.deg2rad(10), 0],
        [2*np.pi/3, np.deg2rad(10), 0],
        [4*np.pi/3, np.deg2rad(10), 0],
    ], (tx_position_without_sectors.shape[0], 1))

    # ------------------ Scene configs ------------------
    scene_cfgs = {
        "mmWave": {
            "carrier_frequency": 28e9,
            "bandwidth": 400e6,
            "power": 55,
            "noise_power": -100,
            "number_rbs": 264,
            "subcarrier_spacing": 60e3,
            "only_los": False,
            "enable_beam_iso_comparison": False,
            "enable_save_paths": True,
            "enable_paths": True,
            "enable_save_images": True,
            "cam_tx_position": [0, 0, 2000.0],
            "tx_array": PlanarArray(
                num_rows=4, num_cols=4,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="tr38901", polarization="V",
            ),
            "rx_array": PlanarArray(
                num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V",
            ),
            "limit_number_ues": None,
            "tx_position": tx_positions_with_sectors,
            "tx_orientation": tx_orientation,
            "enable_beamforming": True,
            "worstcase": True,
        },
        "sub6": {
            "carrier_frequency": 3.5e9,
            "bandwidth": 100e6,
            "power": 49,
            "noise_power": -100,
            "number_rbs": 135,
            "subcarrier_spacing": 15e3,
            "only_los": False,
            "enable_beam_iso_comparison": False,
            "enable_save_paths": True,
            "enable_paths": True,
            "enable_save_images": True,
            "cam_tx_position": [0, 0, 2000.0],
            "tx_array": PlanarArray(
                num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V",
            ),
            "rx_array": PlanarArray(
                num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V",
            ),
            "limit_number_ues": None,
            "tx_position": tx_position_without_sectors,
            "tx_orientation": np.zeros_like(tx_position_without_sectors),
            "enable_beamforming": False,
            "worstcase": True,
        },
    }

    pos_sources = [
        ("real", "x_real", "y_real"),
        ("central_offline", "x_central_offline", "y_central_offline"),
        ("central_online", "x_central_online", "y_central_online"),
        ("federated", "x_federated", "y_federated"),
    ]

    def p3(x, y, z): return [float(x), float(y), float(z)]

    def compute_throughput_vector(scene, p_solver, cfg, rx_pos_xyz):
        scene.get("rx_tmp").position = rx_pos_xyz
        paths = p_solver(
            scene=scene, max_depth=1 if cfg["only_los"] else 4,
            los=True, specular_reflection=True,
            diffuse_reflection=False, refraction=True,
            synthetic_array=True, seed=seed,
        )
        cir_values, _ = paths.cir()
        cir_complex = tf.complex(cir_values[0], cir_values[1])
        cir_complex = tf.cast(cir_complex, dtype=tf.complex64)
        if cfg["enable_beamforming"]:
            tx_positions_np = np.array(
                [np.array(scene.get(f"tx_{i}").position) for i in range(cfg["tx_position"].shape[0])]
            )
            rx_positions_np = np.array([np.array(scene.get("rx_tmp").position)])
            tx_vectors = compute_steering_vector(tx_positions_np, rx_positions_np, cfg["tx_array"])
            xchan = apply_precoder(cir_complex, tx_vectors)
        else:
            xchan = cir_complex
        thr_tx_rx = calc_thr_per_antenna(
            xchan, cfg["power"], cfg["noise_power"], cfg["bandwidth"],
            is_beamformed=cfg["enable_beamforming"], worstcase=True,
        )
        thr = np.array(thr_tx_rx).squeeze()
        if cfg["enable_beamforming"]:
            return np.max(thr.reshape(-1, 3), axis=1)
        else:
            return thr

    # ------------------ Build scenes ------------------
    scenes, solvers, cameras = {}, {}, {}
    for band_name, cfg in scene_cfgs.items():
        print(f"[INFO] Loading scene for band: {band_name}")
        scene = load_scene(scenario_file, merge_shapes=True)
        scene.tx_array = cfg["tx_array"]
        scene.rx_array = cfg["rx_array"]
        scene.frequency = cfg["carrier_frequency"]
        scene.bandwidth = cfg["bandwidth"]
        for tx_id in range(cfg["tx_position"].shape[0]):
            scene.add(Transmitter(
                name=f"tx_{tx_id}",
                position=cfg["tx_position"][tx_id],
                orientation=cfg["tx_orientation"][tx_id],
                display_radius=tx_display_radius,
            ))
        scene.add(Receiver(name="rx_tmp", position=p3(0, 0, user_height), display_radius=rx_display_radius))
        my_cam = Camera(position=cfg["cam_tx_position"], look_at=[0, 0, 0])
        cameras[band_name] = my_cam
        scenes[band_name] = scene
        solvers[band_name] = PathSolver()
        print(f"[INFO] Scene {band_name} loaded successfully")

    # ------------------ Main loop ------------------
    out_rows = []
    img_dir = PathlibPath("img")
    img_dir.mkdir(exist_ok=True)

    for idx, r in df.iterrows():
        global_idx = start + idx
        if global_idx % 50 == 0:
            print(f"[INFO] Processing datapoint {global_idx+1}/{n_total} (user_id={r['id_user']})")

        row = {"row_idx": int(global_idx), "id_user": int(r["id_user"])}
        for src_name, colx, coly in pos_sources:
            rx_xyz = p3(r[colx], r[coly], user_height)
            thr_mm = compute_throughput_vector(scenes["mmWave"], solvers["mmWave"], scene_cfgs["mmWave"], rx_xyz)
            thr_s6 = compute_throughput_vector(scenes["sub6"], solvers["sub6"], scene_cfgs["sub6"], rx_xyz)
            for i in range(tx_position_without_sectors.shape[0]):
                row[f"{src_name}_tx{i}_mmWave"] = float(thr_mm[i])
            for i in range(tx_position_without_sectors.shape[0]):
                row[f"{src_name}_tx{i}_sub6"] = float(thr_s6[i])

            if global_idx % IMG_EVERY == 0:
                try:
                    scene_pic = scenes["mmWave"].render(
                        camera=cameras["mmWave"],
                        num_samples=512,
                        resolution=(1024, 768),
                        return_bitmap=True,
                    )
                    scene_pic = np.rot90(scene_pic, k=-1)
                    plt.imshow(scene_pic)
                    plt.axis("off")
                    plt.savefig(
                        img_dir / f"scene_{global_idx}_{src_name}_{r['id_user']}.png",
                        bbox_inches="tight",
                        pad_inches=0
                    )
                    plt.close()
                except Exception as e:
                    print(f"[WARN] Failed to save image for datapoint {global_idx}: {e}")

        out_rows.append(row)

    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    print(f"[INFO] Segment done: {out_csv}")

# ------------------ ENTRYPOINT ------------------
if "--start" in sys.argv and "--end" in sys.argv:
    s = int(sys.argv[sys.argv.index("--start") + 1])
    e = int(sys.argv[sys.argv.index("--end") + 1])
    run_segment(s, e)
elif "--combine" in sys.argv:
    combine_blocks()
else:
    run_blocks()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import relevant components from Sionna RT
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    Camera,
    PathSolver,
)
import tensorflow as tf
import pandas as pd
from utils import (
    compute_steering_vector,
    apply_precoder,
    calc_thr_per_antenna,
)

scenario_file = "./blender_scenario/ufg.xml"
user_location_filename = "users_location.csv"

user_height = 1.5
seed = 42
tx_display_radius = 5
rx_display_radius = 5

tx_position_without_sectors = np.array(
    [
        [-48.5, 127.8, 50],  # BS 1 position
        [-48.5, 157.8, 50],  # BS 2 position
        [-48.5, 167.8, 50],  # BS 3 position
        [-48.5, 177.8, 50],  # BS 4 position
        [-48.5, 187.8, 50],  # BS 5 position
        [-48.5, 197.8, 50],  # BS 6 position
        [-48.5, 117.8, 50],  # BS 7 position
        [-48.5, 129.8, 50],  # BS 8 position
    ]
)
origin_positions = np.array(
    [
        [0, 0, 0],  # Center UPA
        [0.5, 0, 0],  # Right UPA
        [
            0.5 / 2,
            (0.5 * np.sqrt(3)) / 2,
            0,
        ],  # Left UPA
    ]
)
tx_positions_with_sectors = np.array(
    [origin_positions + tx_pos for tx_pos in tx_position_without_sectors]
).reshape(-1, 3)

tx_orientation = np.tile(
    [
        [
            0,
            np.deg2rad(10),  # Downtilt angle in radians
            0,
        ],  # Center UPA: 0° azimuth, 10° downtilt
        [
            2 * np.pi / 3,
            np.deg2rad(10),
            0,
        ],  # Right UPA: 120° azimuth, 10° downtilt
        [
            4 * np.pi / 3,
            np.deg2rad(10),
            0,
        ],  # Left UPA: 240° azimuth, 10° downtilt
    ],
    (tx_position_without_sectors.shape[0], 1),  # Seven BSs with 3 sectors each
)


def p3(x, y, z):
    """Returns three Python floats (not numpy) for Mitsuba/Dr.Jit."""
    return [float(x), float(y), float(z)]


scene_configs = {
    "mmwave": {
        "carrier_frequency": 28e9,  # 28 GHz
        "bandwidth": 400e6,  # 400 MHz
        "power": 55,  # Transmit power in dBm
        "noise_power": -100,  # Noise power in dBm
        "number_rbs": 264,  # Number of resource blocks
        "subcarrier_spacing": 60e3,  # Subcarrier spacing in Hz
        "only_los": False,  # Consider all paths
        "enable_beam_iso_comparison": False,  # Compare beamforming throughput per antenna
        "enable_save_paths": True,
        "enable_paths": True,
        "enable_save_images": True,
        "cam_tx_position": [0, 0, 2000.0],
        "tx_array": PlanarArray(
            num_rows=4,
            num_cols=4,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        ),
        "rx_array": PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        ),
        "limit_number_ues": None,  # Limit number of UEs or None in case you want all UEs
        "tx_position": tx_positions_with_sectors,
        "tx_orientation": tx_orientation,
        "enable_beamforming": True,  # Enable beamforming
        "worstcase": True,  # Enable worst-case interference scenario (all TXs pointing in the Receiver direction)
    },
    "sub6": {
        "carrier_frequency": 3.5e9,  # 3.5 GHz
        "bandwidth": 100e6,  # 100 MHz
        "power": 49,  # Transmit power in dBm
        "noise_power": -100,  # Noise power in dBm
        "number_rbs": 135,  # Number of resource blocks
        "subcarrier_spacing": 15e3,  # Subcarrier spacing in Hz
        "only_los": False,  # Consider all paths
        "enable_beam_iso_comparison": False,  # Compare beamforming throughput per antenna
        "enable_save_paths": True,
        "enable_paths": True,
        "enable_save_images": True,
        "cam_tx_position": [0, 0, 2000.0],
        "tx_array": PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        ),
        "rx_array": PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        ),
        "limit_number_ues": None,  # Limit number of UEs or None in case you want all UEs
        "tx_position": tx_position_without_sectors,
        "tx_orientation": np.zeros_like(tx_position_without_sectors),
        "enable_beamforming": False,  # Enable beamforming
        "worstcase": True,  # Enable worst-case interference scenario (all TXs pointing in the Receiver direction)
    },
}
# Reading CSV file with user locations
users_df = pd.read_csv(user_location_filename, usecols=["id", "x", "y", "o"])

# bounding box center of the SUMO scenario (assumes equal number of samples per id)
xmin, ymin = users_df[["x", "y"]].min()
xmax, ymax = users_df[["x", "y"]].max()
cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

# alignment parameters to match the Blender/Mitsuba coordinate system
MAP_ROT_DEG = 0  # degrees
OFFSET_XY = (-188.0, -10.0)  # meters

for scene_id, scene_config in tqdm(scene_configs.items(), desc="Scenario"):
    folder_path = Path(f"data/scene_{scene_id}")
    folder_path.mkdir(parents=True, exist_ok=True)

    my_cam = Camera(position=scene_config["cam_tx_position"], look_at=[0, 0, 0])

    scene = load_scene(scenario_file, merge_shapes=True)

    # Configure antenna array for all transmitters
    scene.tx_array = scene_config["tx_array"]

    # Configure antenna array for all receivers
    scene.rx_array = scene_config["rx_array"]

    scene.frequency = scene_config["carrier_frequency"]
    scene.bandwidth = scene_config["bandwidth"]

    # Create transmitter
    orientations = scene_config["tx_orientation"]
    tx_positions = scene_config["tx_position"]
    for tx_id in range(scene_config["tx_position"].shape[0]):
        tx = Transmitter(
            name=f"tx_{tx_id}",
            position=tx_positions[tx_id],
            orientation=orientations[tx_id],
            display_radius=tx_display_radius,
        )
        # Add transmitter instance to scene
        scene.add(tx)

    # Instantiate a path solver
    p_solver = PathSolver()

    # Computing propagation paths for each user (and creating receivers)
    for step in tqdm(
        range(users_df["id"].value_counts()[0]), desc="Steps", leave=False
    ):
        ues_available = (
            users_df["id"].unique()
            if scene_config["limit_number_ues"] is None
            else range(scene_config["limit_number_ues"])
        )
        for user in ues_available:
            user_df = users_df[users_df["id"] == user]

            # raw (x,y) from SUMO
            sx = float(user_df["x"].values[step])
            sy = float(user_df["y"].values[step])

            # 1) recenter: corner -> center
            x0 = sx - cx
            y0 = sy - cy

            # 2) rotate in plane (if needed)
            th = np.deg2rad(MAP_ROT_DEG)
            xr = np.cos(th) * x0 - np.sin(th) * y0
            yr = np.sin(th) * x0 + np.cos(th) * y0

            # 3) global offset for fine alignment between SUMO and Sionna coordinate centers
            xr += OFFSET_XY[0]
            yr += OFFSET_XY[1]

            rx_position = [float(xr), float(yr), float(user_height)]

            if step == 0:
                rx = Receiver(
                    name=f"rx_{user}",
                    position=p3(xr, yr, user_height),
                    display_radius=rx_display_radius,
                )

                scene.add(rx)
            else:
                scene.get(f"rx_{user}").position = p3(xr, yr, user_height)

        # Compute propagation paths
        if scene_config["enable_paths"]:
            paths = p_solver(
                scene=scene,
                max_depth=1 if scene_config["only_los"] else 4,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=True,
                synthetic_array=True,
                seed=seed,
            )

            if scene_config["enable_save_paths"]:
                tx_positions = np.array(
                    [
                        np.array(scene.get(f"tx_{tx_id}").position)
                        for tx_id in range(scene_config["tx_position"].shape[0])
                    ]
                )
                rx_positions = np.array(
                    [
                        np.array(scene.get(f"rx_{user}").position)
                        for user in ues_available
                    ]
                )
                tx_vectors = compute_steering_vector(
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    tx_array=scene_config["tx_array"],
                )
                cir_values, cir_delays = paths.cir()
                cir_complex = tf.complex(cir_values[0], cir_values[1])
                cir_complex = tf.cast(cir_complex, dtype=tf.complex64)
                effective_cir = apply_precoder(cir_complex, tx_vectors)
                thr_tx_rx = calc_thr_per_antenna(
                    (
                        effective_cir
                        if scene_config["enable_beamforming"]
                        else cir_complex
                    ),
                    scene_config["power"],
                    scene_config["noise_power"],
                    scene_config["bandwidth"],
                    is_beamformed=scene_config["enable_beamforming"],
                    worstcase=scene_config["worstcase"],
                )
                np.savez_compressed(
                    f"data/scene_{scene_id}/step_{step}.npz",
                    thr_tx_rx=thr_tx_rx,
                    channels=(
                        effective_cir.numpy()
                        if scene_config["enable_beamforming"]
                        else cir_complex.numpy()
                    ),
                )
                if scene_config["enable_beam_iso_comparison"]:
                    cir_complex = tf.complex(cir_values[0], cir_values[1])
                    cir_complex = tf.cast(cir_complex, dtype=tf.complex64)
                    thr_omni_tx_rx = calc_thr_per_antenna(
                        cir_complex,
                        scene_config["power"],
                        scene_config["noise_power"],
                        scene_config["bandwidth"],
                        is_beamformed=False,
                        worstcase=scene_config["worstcase"],
                    )
                    max_iso = tf.reduce_max(thr_omni_tx_rx, axis=1)
                    max_beam = tf.reduce_max(thr_tx_rx, axis=1)

                    print(
                        f"From {thr_tx_rx.shape[0]} RXs, Beamforming is better than omni in: {np.sum(max_beam > max_iso)}"
                    )
        if scene_config["enable_save_images"]:
            scene_pic = scene.render(
                camera=my_cam,
                paths=paths if scene_config["enable_paths"] else None,
                num_samples=512,
                resolution=(1024, 768),
                return_bitmap=True,
            )

            # rotate 90° clockwise
            scene_pic = np.rot90(scene_pic, k=-1)

            plt.imshow(scene_pic)
            plt.axis("off")
            plt.savefig(f"{str(folder_path)}/step_{step}.png")
            plt.close()

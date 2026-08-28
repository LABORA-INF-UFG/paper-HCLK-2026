# Replication Package — Network-Wide Association Optimization with Learned Mobility Prediction for Minimizing Breaks in Presence in Mobile Augmented Reality

This repository contains the complete pipeline to reproduce the results reported in the paper "Network-Wide Association Optimization with Learned Mobility Prediction for Minimizing Breaks in Presence in Mobile Augmented Reality," by H. P. Romualdo, C. V. Nahum, L. S. Fraga, and K. V. Cardoso.

---

## Repository Structure

```
paper-HCLK-2026/
├── data/                        # Datasets (mobility, channel, BS positions)
├── sumo/                        # Mobility trace generation (SUMO)
├── sionna/                      # Channel simulation (Sionna, runs on WSL)
├── placement/                   # BS placement optimization (NSGA-II)
├── model/                       # ESN model class
├── dataset_best_windows_metadata.py
├── dataset_feature_engineering.py
├── evaluate_bip_methodology.py
├── export_val_predictions.py
├── split_throughput_dataset.py
├── train_central_offline.py
├── train_central_online.py
├── train_federated_online.py
```

---

## Pipeline Overview

The pipeline has five sequential stages. Pre-computed outputs for each stage are provided in `data/` so that any stage can be run independently.

```
[1] SUMO          →  dataset.csv               (mobility traces)
[2] Feature Eng.  →  derived_features_full.csv (kinematic features)
[3] Sionna (WSL)  →  throughput_users_50.csv   (channel simulation)
[4] ESN Training  →  model/*/                  (centralized/federated)
[5] Evaluation    →  paper figures & tables
```

---

## Requirements

### Python environment (stages 1, 2, 4, 5)

Python 3.9 or later. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib pulp
```

### SUMO — Simulation of Urban MObility (stage 1)

SUMO is required only to regenerate the mobility traces from scratch. The pre-computed output (`data/dataset.csv` and `sumo/fcd.xml`) is already provided.

**Installation on Windows:**
1. Download the SUMO installer from https://sumo.dlr.de/docs/Downloads.php
2. Run the installer and add SUMO to the system PATH
3. Verify: `sumo --version`

**Installation on Ubuntu / WSL:**
```bash
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

### Sionna — Ray-Tracing Channel Simulation (stage 3)

Sionna requires a Linux environment with GPU support. On Windows 11, use **WSL 2** with NVIDIA CUDA.

#### Setting up WSL 2 on Windows 11

1. Enable WSL 2 (run in PowerShell as Administrator):
   ```powershell
   wsl --install
   wsl --set-default-version 2
   ```
2. Install Ubuntu 22.04 from the Microsoft Store
3. Install NVIDIA drivers for WSL: https://developer.nvidia.com/cuda/wsl
4. Verify GPU access inside WSL:
   ```bash
   nvidia-smi
   ```

#### Installing Sionna inside WSL

```bash
# Inside WSL Ubuntu terminal, from the repository root
cd sionna

# Install pipenv
pip install pipenv

# Create environment and install dependencies (see Pipfile)
pipenv install

# Activate environment
pipenv shell
```

> **Note:** Sionna requires TensorFlow 2.x with GPU support. The `Pipfile` in `sionna/` specifies the exact versions used in this work.

---

## Stage-by-Stage Execution

### Stage 1 — Mobility Trace Generation (SUMO)

Generates 50-user pedestrian mobility traces over a 1-hour simulation of the UFG campus scenario.

```bash
cd sumo/
python 50_users_1_hour_by_sec.py
# Output: sumo/fcd.xml  →  copy to data/dataset.csv
```

The OpenStreetMap scenario (`osm.pedestrians.rou.xml`) and trip definitions (`trips.trips.xml`) are provided. The `sumo/data/` directory contains the SUMO network and configuration files.

Pre-computed output: `data/dataset.csv`

---

### Stage 2 — Feature Engineering

Extracts 15 kinematic and angular features from the raw mobility traces.

```bash
python dataset_feature_engineering.py \
    --input data/dataset.csv \
    --output data/derived_features_full.csv
```

Output columns include: `z_x, z_y, z_vx, z_vy, z_speed, z_ax, z_ay, z_accel, z_turn_rate, cos_heading, sin_heading, cos_o, sin_o, cos_gaze_move, sin_gaze_move`.

Pre-computed output: `data/derived_features_full.csv`

---

### Stage 3 — Channel Simulation (Sionna, WSL)

Computes per-user, per-BS throughput for sub-6 GHz and mmWave bands using 3GPP-compliant ray tracing over the Blender 3D scenario of the UFG campus.

```bash
# Run inside WSL with Sionna environment activated
cd sionna/
bash run_throughput_segments.sh
# Or for a specific user range:
python compute_throughput.py --start 0 --end 1000
```

The `blender_scenario/` directory contains the 3D scene used for ray tracing. The `osm.poly.xml` file defines the geographic boundary.

Pre-computed outputs (Windows `data/` directory):
- `data/throughput_users_50.csv` — 50-user scenario (default)
- `data/throughput_users_{10,20,30,40}.csv` — scalability evaluation
- `data/throughput_stations_{BSIDs}.csv` — infrastructure density evaluation

---

### Stage 4 — ESN Training

Three paradigms are evaluated. All use an 80/20 temporal split per user. The 20% holdout is identical across paradigms, ensuring a fair comparison.

**Centralized Offline**
```bash
python train_central_offline.py \
    --csv data/derived_features_full.csv \
    --out model/centralized_offline \
    --steps_ahead 3
```

**Centralized Online**
```bash
python train_central_online.py \
    --csv data/derived_features_full.csv \
    --ru_csv data/ru_positions.csv \
    --out model/centralized_online \
    --steps_ahead 3 --rounds 100
```

**Federated Online**
```bash
python train_federated_online.py \
    --csv data/derived_features_full.csv \
    --ru_csv data/ru_positions.csv \
    --out model/federated \
    --steps_ahead 3 --rounds 100 --num_rus 9
```

Reservoir hyperparameters (fixed across all paradigms): `Nr=600`, `leak=0.3`, `spectral_radius=0.9`, `sparsity=0.1`, `H=3s`. The RLS readout differs by paradigm: the centralized-offline paradigm uses no forgetting (`λ_RLS=1`, `α_RLS=1e-4`), since it fits the entire history in a single batch, while the centralized-online and federated-online paradigms use `λ_RLS=0.99`, `α_RLS=1e-3` to track non-stationarity across rounds.

---

### Stage 5 — Evaluation and Paper Results

```bash
# Export mobility predictions (needed by the optimizer)
python export_val_predictions.py \
    --csv data/derived_features_full.csv \
    --models_dir model/ \
    --out data/predictions/

# Run ILP-based association optimization and compute QoE metrics
python evaluate_bip_methodology.py \
    --throughput data/throughput_users_50.csv \
    --predictions data/predictions/ \
    --ru_csv data/ru_positions.csv
```

---

## BS Placement

The 9-BS deployment was obtained by solving a bi-objective NSGA-II optimization minimizing coverage gaps and spatial overlap. To reproduce:

```bash
cd placement/
python bs_placement.py
python bs_placement_plotting.py
```

Pre-computed result: `data/ru_positions.csv`

---

## Large Files and Git LFS

The following files exceed 50 MB and are tracked with **Git LFS**:

| File | Size | Description |
|---|---|---|
| `data/derived_features_full.csv` | ~59 MB | Feature-engineered mobility dataset |
| `sumo/fcd.xml` | ~21 MB | Raw SUMO floating car data output |
| `sumo/osm.pedestrians.rou.xml` | ~8 MB | SUMO pedestrian route definitions |

To clone the repository with LFS content:
```bash
git lfs install
git clone https://github.com/LABORA-INF-UFG/paper-HCLK-2026
```

---

## Citation

```bibtex
@article{romualdo2026mar,
  author = {Romualdo, Hudson de Paula and Nahum, Cleverson V. and Fraga, Luciano de Souza and Cardoso, Kleber V.},
  title  = {Network-Wide Association Optimization with Learned Mobility Prediction for Minimizing Breaks in Presence in Mobile Augmented Reality},
  year   = {2026}
}
```

---

## Contact

Hudson de Paula Romualdo — hudson_romualdo@discente.ufg.br  
LABORA — Laboratory of Computer Networks and Multimedia, UFG, Brazil

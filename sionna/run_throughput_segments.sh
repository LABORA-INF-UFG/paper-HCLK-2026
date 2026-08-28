#!/bin/bash
# run_segments_safe.sh
# Runs each block in an isolated shell process (not inside the same Python process)

source ~/sionna_env/bin/activate

rm -f throughput_users_part_*.csv throughput_users.csv

echo "[RUN] Segment 0–1000"
bash -c "python compute_throughput_by_block.py --start 0 --end 1000"

echo "[RUN] Segment 1000–2000"
bash -c "python compute_throughput_by_block.py --start 1000 --end 2000"

echo "[RUN] Segment 2000–3000"
bash -c "python compute_throughput_by_block.py --start 2000 --end 3000"

echo "[RUN] Combining"
bash -c "python compute_throughput_by_block.py --combine"

echo "[OK] Done."

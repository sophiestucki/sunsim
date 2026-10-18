#!/bin/bash
# Step 3 (one job, after all the sets): table of the MAD / RMS of every set (sweep_stats.csv) and the figures
# (heatmaps I_th x A_th per T_sp and I_th x T_sp per A_th, RMS against every parameter; RV, SORCE, TSIS-1):
#   bash collect_stats.sh
PIPELINES=/data/spotless/scratch/sstucki/starsim/pipelines
source "${PIPELINES}/settings.sh"
python -m sunsim.observations collect "${BASE}/sets" --out "${BASE}/results"

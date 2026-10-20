#!/bin/bash
# Step 5 (one job, after all the sets): table of the MAD / RMS of every set (sweep_stats.csv) and the figures (heatmaps
# of I_th against B_sat, one panel per T_sp):
#   bash collect_stats_fraction.sh
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_fraction.sh"
python -m sunsim.observations collect "$SETS" --out "${BASE}/results_${TAG}"

#!/bin/bash
# Step 3 (one job per kind, after all its sets): table of the MAD / RMS of every set (sweep_stats.csv) and the figures:
#   bash collect_stats_45_dumusque_mu.sh <plage|fraction>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
KIND=$1
source "${PIPELINES}/settings_45_dumusque_mu.sh"
python -m sunsim.observations collect "$SETS" --out "${BASE_DMU}/results_${TAG}"

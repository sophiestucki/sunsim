#!/bin/bash
# Step 5 (one job per kind, after all its sets): table of the MAD / RMS of every set (sweep_stats.csv) and the figures:
#   bash collect_stats_720.sh <plage|fraction>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
KIND=$1
source "${PIPELINES}/settings_720.sh"
python -m sunsim.observations collect "$SETS" --out "${BASE720}/results_${TAG}"

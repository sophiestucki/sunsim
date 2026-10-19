#!/bin/bash
# Step 5 (one job, after all the sets): table of the MAD / RMS of every set (sweep_stats.csv) and the figures:
#   bash collect_stats_4096.sh
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_4096.sh"
python -m sunsim.observations collect "$SETS" --out "${BASE}/results_${TAG}"

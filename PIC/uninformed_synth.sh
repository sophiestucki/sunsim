#!/bin/bash
# Step 3 (one job, after the training): 3 months of synthetic masks at the minimum and at the maximum, 3
# realizations each, and the statistics of SDO and of the synthetic masks: ${BASE_U}/synthetic
# (bash uninformed_synth.sh <workers>)
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_uninformed.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/uninformed_decade.py" synth --out "$BASE_U" --min "$SYN_MIN" --max "$SYN_MAX" --workers "$1"

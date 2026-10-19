#!/bin/bash
# Step 3 (one job, after all the blocks): the same sweep over the whole period with python -m sunsim.sdo: the days
# still missing are done, then filling_factors_N60.npz, summary.csv and parameters.txt of every set, and sets.csv:
#   bash aggregate_4096.sh <workers>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_4096.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python -m sunsim.sdo --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" --size "$SIZE" \
       --noise-th $NOISE_VALUES --I-th $I_VALUES --A-th $A_VALUES --format none --n-rings "$N_RINGS" --workers "$1"

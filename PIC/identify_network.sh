#!/bin/bash
# Step 2 of sweep_network.dag (N_BLOCKS jobs): identification of one block of days at 4096 px (45 s magnetograms) for
# every set of parameters_network.txt at once and the noise diagnostics (identify_network.py), in ${IDENT}:
#   bash identify_network.sh <workers> <block>
# A block can be started again: the days done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_network.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/identify_network.py" --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" \
       --n-rings "$N_RINGS" --params "$PARAMS" --diag-noise-th "$DIAG_NOISE_TH" --diag-B-th $DIAG_B_TH \
       --diag-I-th "$DIAG_I_TH" --diag-B-sat "$DIAG_B_SAT" --magnetogram "$MAGNETOGRAM" \
       --block "$2" --n-blocks "$N_BLOCKS" --workers "$1"

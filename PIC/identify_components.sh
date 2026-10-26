#!/bin/bash
# Step 2 (N_BLOCKS jobs): identification of one block of days at 4096 px for all the sets of thresholds of
# parameters_components.txt at once, each with faculae='plage' and faculae='plage+network' (every day read once),
# cell filling factors of N_RINGS rings in ${IDENT}/<set label>/ and ${IDENT}/<set label>_network/:
#   bash identify_components.sh <workers> <block>
# A block can be started again: the days done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/identify_days.py" --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" \
       --size "$SIZE" --n-rings "$N_RINGS" --noise-th $NOISE_VALUES --I-th $I_VALUES --A-th $A_VALUES \
       --faculae plage plage+network --magnetogram "$MAGNETOGRAM" --block "$2" --n-blocks "$N_BLOCKS" --workers "$1"

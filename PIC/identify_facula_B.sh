#!/bin/bash
# Step 2 of sweep_facula_B.dag (N_BLOCKS jobs): identification of one block of days at 4096 px (45 s magnetograms) for
# the sets of thresholds of parameters_facula_B.txt at once (identify_days.py --params; B_fc and Temp are not
# thresholds), cell filling factors of N_RINGS rings in ${IDENT}/<set label>/:
#   bash identify_facula_B.sh <workers> <block>
# A block can be started again: the days done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_facula_B.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/identify_days.py" --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" \
       --size "$SIZE" --n-rings "$N_RINGS" --params "$PARAMS" --columns "$PARAM_COLUMNS" --faculae "$FACULAE" \
       --magnetogram "$MAGNETOGRAM" --block "$2" --n-blocks "$N_BLOCKS" --workers "$1"

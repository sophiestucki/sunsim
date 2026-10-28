#!/bin/bash
# Step 2 of sweep_noise.dag (N_BLOCKS jobs per kind): identification of one block of days at 4096 px (45 s
# magnetograms) for the sets of the lines of the parameter file of KIND at once (every day read once; B_th with its
# noise_th, identify_days.py --params), cell filling factors of N_RINGS rings in ${IDENT}/<set label>/:
#   bash identify_noise.sh <plage|fraction> <workers> <block>
# A block can be started again: the days done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
KIND=$1
source "${PIPELINES}/settings_noise.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/identify_days.py" --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" \
       --size "$SIZE" --n-rings "$N_RINGS" --params "$PARAMS" --columns "$PARAM_COLUMNS" --faculae "$FACULAE" \
       --magnetogram "$MAGNETOGRAM" --block "$3" --n-blocks "$N_BLOCKS" --workers "$2"

#!/bin/bash
# Step 2 (N_BLOCKS jobs per kind): identification of one block of days at 4096 px with the 720 s magnetograms, for all
# the sets of thresholds of the parameter file of KIND at once (every day read once), cell filling factors of N_RINGS
# rings in ${IDENT}/<set label>/:
#   bash identify_720.sh <plage|fraction> <workers> <block>
# A block can be started again: the days done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
KIND=$1
source "${PIPELINES}/settings_720.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/identify_days.py" --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" \
       --size "$SIZE" --n-rings "$N_RINGS" --noise-th $NOISE_VALUES --I-th $I_VALUES $X_OPT $X_VALUES \
       --faculae "$FACULAE" --magnetogram "$MAGNETOGRAM" --block "$3" --n-blocks "$N_BLOCKS" --workers "$2"

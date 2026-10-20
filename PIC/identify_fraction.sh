#!/bin/bash
# Step 2 (N_BLOCKS jobs): identification of one block of days at 4096 px with facular fractions, for all the sets
# (noise_th, I_th, B_sat) of the parameter file at once (every day read once; the sets of the same noise_th and I_th
# share the classification), cell filling factors of 60 rings in ${IDENT}/<set label>/:
#   bash identify_fraction.sh <workers> <block>
# A block can be started again: the days done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_fraction.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/identify_days.py" --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" \
       --size "$SIZE" --n-rings "$N_RINGS" --noise-th $NOISE_VALUES --I-th $I_VALUES \
       --faculae fraction --B-sat $BSAT_VALUES --block "$2" --n-blocks "$N_BLOCKS" --workers "$1"

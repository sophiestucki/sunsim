#!/bin/bash
# Step 1 (N_BLOCKS jobs): the SDO masks of one block of days of the decade at 1024 px (uninformed_decade.py
# maps-local, from the images of SDO_DATA):  bash uninformed_maps.sh <workers> <block>
# A block can be started again: the days done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_uninformed.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/uninformed_decade.py" maps-local --data "$SDO_DATA" --out "$BASE_U" --start "$START" \
       --end "$END" --block "$2" --n-blocks "$N_BLOCKS" --workers "$1"

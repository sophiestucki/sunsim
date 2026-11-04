#!/bin/bash
# Step 2 (one job): the filling factors of all the days in one folder per set of thresholds (${POET_OUT}/ident/<set
# label>/) and the epoch of every exposure (${POET_OUT}/exposures.csv). The days not done are left out
# (${POET_OUT}/incomplete_days.txt): run their poet_day.sh again, then this.
#   bash poet_collect.sh
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_poet.sh"

python "${PIPELINES}/poet_epochs.py" collect --times "$POET_TIMES" --out "$POET_OUT" --params "$PARAMS" \
       --start "$POET_START" --size "$SIZE" --n-rings "$N_RINGS"

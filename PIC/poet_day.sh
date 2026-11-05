#!/bin/bash
# Step 1 (one job per day of poet_days.txt): the HMI epochs closest to the PoET exposures of the day, downloaded from
# JSOC, identified with every set of thresholds of parameters_poet.txt and deleted right after:
#   bash poet_day.sh <threads> <YYYY-MM-DD>
# A day can be started again: the epochs done are skipped. Exit code always 0: DAGMan removes every other day of the
# cluster as soon as one job fails. A failed day has no done.json (incomplete_days.txt of collect) or failed > 0 in it.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_poet.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

python "${PIPELINES}/poet_epochs.py" day --times "$POET_TIMES" --date "$2" --out "$POET_OUT" --params "$PARAMS" \
       --size "$SIZE" --n-rings "$N_RINGS" --max-dt "$MAX_DT" --tmp "${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
status=$?
[ $status -ne 0 ] && echo "ERROR: $2 exit code $status (run the day again)"
exit 0

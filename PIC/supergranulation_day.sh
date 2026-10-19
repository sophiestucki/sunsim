#!/bin/bash
# One day: the HMI epochs closest to its HARPS-N exposures (identification + 720 s dopplergram) and the 45 s
# dopplergrams (p modes), downloaded from JSOC and deleted right after processing (only small products are kept):
#   bash supergranulation_day.sh <threads> <YYYY-MM-DD>
# A day can be started again: what is done is skipped. Exit code 1 if an epoch or a frame failed (run it again).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_supergranulation.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1
day=$2

python "${PIPELINES}/supergranulation.py" day --date "$day" --out "$SG_OUT" --harpsn "$HARPSN" \
       --size "$SG_SIZE" --n-rings "$SG_RINGS" $SG_THRESHOLDS $SG_PMODES --workers "$SG_WORKERS" \
       --tmp "${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"

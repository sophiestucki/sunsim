#!/bin/bash
# One day: the HMI epochs closest to its NEID exposures (identification) and the 45 s dopplergrams around them,
# corrected (Haywood et al. 2016, eq. 10 and 13) and projected on the cells, and the JPL velocity of the Sun at Kitt
# Peak; the FITS are downloaded from JSOC and deleted right after processing (only small products are kept):
#   bash supergranulation_haywood_neid_day.sh <threads> <YYYY-MM-DD>
# A day can be started again: what is done is skipped. Exit code 1 if an epoch or a frame failed (run it again).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_supergranulation_haywood_neid.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1
day=$2

python "${PIPELINES}/supergranulation_haywood_neid.py" day --neid "$NEID" --date "$day" --out "$SG_OUT" \
       --size "$SG_SIZE" --n-rings "$SG_RINGS" $SG_THRESHOLDS --window "$SG_WINDOW" --mu-min "$SG_MU_MIN" \
       --workers "$SG_WORKERS" --tmp "${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"

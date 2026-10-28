#!/bin/bash
# Step 5 of sweep_noise.dag (one job per kind, after all its sets): table of the MAD / RMS of every set
# (sweep_stats.csv) and the figures, then the disc filling factors of every set against SATIRE-S on the days with a
# SATIRE mask (satire_ff.csv; SATIRE_FF of settings_noise.sh, made by satire_ff.py satire):
#   bash collect_noise.sh <plage|fraction>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
KIND=$1
source "${PIPELINES}/settings_noise.sh"
out="${BASE_N}/results_${TAG}"
python -m sunsim.observations collect "$SETS" --out "$out"
status=$?
if [ -f "$SATIRE_FF" ]; then
    python "${PIPELINES}/satire_ff.py" compare --ident "$IDENT" --satire "$SATIRE_FF" --out "${out}/satire_ff.csv" || status=1
else
    echo "ERROR: no $SATIRE_FF (satire_ff.py satire): no comparison with SATIRE"
    status=1
fi
exit $status

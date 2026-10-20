#!/bin/bash
# Step 3 (N_JOBS jobs, after all the blocks): python -m sunsim.sdo over the whole period for the sets job, job + N_JOBS,
# ... of the (I_th, B_sat) pairs of the parameter file (all the noise_th): the days still missing are done, then
# filling_factors_N60.npz, summary.csv and parameters.txt of every set, and sets.csv:
#   bash aggregate_fraction.sh <workers> <job> <N_JOBS>
# (the jobs write different set folders; they all add their sets to ${IDENT}/sets.csv, which is not used by the next
# steps)
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_fraction.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

pairs=$(for I in $I_VALUES; do for B in $BSAT_VALUES; do echo "$I $B"; done; done |
        awk -v k="$2" -v n="$3" '(NR - 1) % n == k')
if [ -z "$pairs" ]; then
    echo "job $2 of $3: no set (only $(( $(echo $I_VALUES | wc -w) * $(echo $BSAT_VALUES | wc -w) )) sets)"
    exit 0
fi
status=0
while read -r I B; do
    echo "job $2 of $3: I_th = $I, B_sat = $B"
    python -m sunsim.sdo --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" --size "$SIZE" \
           --noise-th $NOISE_VALUES --I-th "$I" --faculae fraction --B-sat "$B" \
           --format none --n-rings "$N_RINGS" --workers "$1" || status=1
done <<< "$pairs"
exit $status

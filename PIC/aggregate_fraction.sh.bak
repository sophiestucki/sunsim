#!/bin/bash
# Step 3 (N_JOBS jobs): the days still missing and filling_factors_N60.npz, summary.csv, parameters.txt of every set
# (I_th, B_sat), each in its own folder ${IDENT}/<set label>:  bash aggregate_fraction.sh <workers> <job> <N_JOBS>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_fraction.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

pairs=$(for I in $I_VALUES; do for B in $BSAT_VALUES; do echo "$I $B"; done; done |
        awk -v k="$2" -v n="$3" '(NR - 1) % n == k')
if [ -z "$pairs" ]; then
    echo "job $2 of $3: no set"
    exit 0
fi
status=0
while read -r I B; do
    label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$NOISE_VALUES, I_th=$I, B_sat=$B, faculae='fraction')))")
    echo "job $2 of $3: I_th = $I, B_sat = $B -> ${IDENT}/${label}"
    python -m sunsim.sdo --data "$SDO_DATA" --out "${IDENT}/${label}" --start "$START" --end "$END" --size "$SIZE" \
           --noise-th $NOISE_VALUES --I-th "$I" --faculae fraction --B-sat "$B" \
           --format none --n-rings "$N_RINGS" --workers "$1" || status=1
done <<< "$pairs"
exit $status

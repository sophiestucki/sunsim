#!/bin/bash
# Step 3 (N_AGGREGATE jobs per kind): the days still missing and filling_factors_N<N_RINGS>.npz, summary.csv,
# parameters.txt of every set (noise_th, I_th, A_th or B_sat) of the parameter file of KIND, each in its own folder
# ${IDENT}/<set label>; job k does the sets k, k + n_jobs, ...:
#   bash aggregate_720.sh <plage|fraction> <workers> <job> <n_jobs>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
KIND=$1
source "${PIPELINES}/settings_720.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

sets=$(awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {print $1 + 0, $2 + 0, $3 + 0}' "$PARAMS" | sort -u |
       awk -v k="$3" -v n="$4" '(NR - 1) % n == k')
if [ -z "$sets" ]; then
    echo "job $3 of $4: no set"
    exit 0
fi
status=0
while read -r noise I X; do
    label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise, I_th=$I, $X_NAME=$X, faculae='$FACULAE')))")
    echo "job $3 of $4: noise_th = $noise, I_th = $I, $X_NAME = $X -> ${IDENT}/${label}"
    python -m sunsim.sdo --data "$SDO_DATA" --out "${IDENT}/${label}" --start "$START" --end "$END" --size "$SIZE" \
           --noise-th "$noise" --I-th "$I" $X_OPT "$X" --faculae "$FACULAE" --magnetogram "$MAGNETOGRAM" \
           --format none --n-rings "$N_RINGS" --workers "$2" || status=1
done <<< "$sets"
exit $status

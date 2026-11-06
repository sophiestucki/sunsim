#!/bin/bash
# Step 3 of sweep_facula_B.dag (one job, after all the blocks): the days still missing and filling_factors_N<N_RINGS>.npz,
# summary.csv, parameters.txt of every set of thresholds (noise_th, B_th, I_th, A_th) of parameters_facula_B.txt,
# each in its own folder ${IDENT}/<set label>:
#   bash aggregate_facula_B.sh <workers>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_facula_B.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

sets=$(awk -F'[ ,\t]+' 'NF >= 6 && $1 !~ /^#/ {print $1 + 0, $2 + 0, $3 + 0, $4 + 0}' "$PARAMS" | sort -u)
status=0
while read -r noise B I A; do
    label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise, B_th=$B, I_th=$I, A_th=$A, faculae='$FACULAE')))")
    echo "noise_th = $noise, B_th = $B, I_th = $I, A_th = $A -> ${IDENT}/${label}"
    python -m sunsim.sdo --data "$SDO_DATA" --out "${IDENT}/${label}" --start "$START" --end "$END" --size "$SIZE" \
           --noise-th "$noise" --B-th "$B" --I-th "$I" --A-th "$A" --faculae "$FACULAE" --magnetogram "$MAGNETOGRAM" \
           --format none --n-rings "$N_RINGS" --workers "$1" || status=1
done <<< "$sets"
exit $status

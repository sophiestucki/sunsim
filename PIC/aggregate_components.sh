#!/bin/bash
# Step 3 (one job, after all the blocks): for every set (noise_th, I_th, A_th) of parameters_components.txt, the days
# still missing and filling_factors_N<N_RINGS>.npz, summary.csv, parameters.txt of its two identifications (plage,
# plage+network), then the filling factors of its parts in ${PARTS_DIR}/<set label>/<part>/ (split_components.py):
#   bash aggregate_components.sh <workers>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

status=0
while read -r noise I A; do
    label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise, I_th=$I, A_th=$A, faculae='plage')))")
    label_nt=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise, I_th=$I, A_th=$A, faculae='plage+network')))")
    echo "noise_th = $noise, I_th = $I, A_th = $A -> ${IDENT}/${label}, ${IDENT}/${label_nt}"
    python -m sunsim.sdo --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" --size "$SIZE" \
           --noise-th "$noise" --I-th "$I" --A-th "$A" --faculae plage plage+network --magnetogram "$MAGNETOGRAM" \
           --format none --n-rings "$N_RINGS" --workers "$1" || { status=1; continue; }
    python "${PIPELINES}/split_components.py" "${IDENT}/${label}" "${IDENT}/${label_nt}" "${PARTS_DIR}/${label}" \
           --n-rings "$N_RINGS" || status=1
done < <(awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {print $1 + 0, $2 + 0, $3 + 0}' "$PARAMS" | sort -u)
exit $status

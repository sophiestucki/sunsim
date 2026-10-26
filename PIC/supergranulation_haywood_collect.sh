#!/bin/bash
# Step 2 (one job, after all the days): the corrected dopplergrams averaged over the window of every exposure,
# simulation of every exposure (maps; maps + dopplergram), comparison with the HARPS-N RV in the rest frame of the Sun
# (rv_raw - JPL), statistics and figures in ${SG_OUT}/<results>, once per variant of SG_VARIANTS:
#   bash supergranulation_haywood_collect.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_supergranulation_haywood.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

status=0
for variant in "${SG_VARIANTS[@]}"; do
    results=${variant%%:*}
    options=${variant#*:}
    echo "==== ${results}: ${options}"
    python "${PIPELINES}/supergranulation_haywood.py" collect --out "$SG_OUT" --start "$SG_START" --conf "$SG_CONF" \
           --mps-folder "$SG_MPS" --mask "$SG_MASK" --interventions "$INTERVENTIONS" --n-rings "$SG_RINGS" \
           $SG_QUALITY --results "$results" $options || status=1
done
exit $status

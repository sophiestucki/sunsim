#!/bin/bash
# Step 2 (one job, after the days): the corrected dopplergrams averaged over the window of every NEID exposure,
# simulation of every exposure (maps; maps + dopplergram), comparison with the NEID RV corrected as the HARPS-N RV
# (rest frame of the Sun, differential extinction), statistics and figures in ${SG_OUT}/<results>, once per variant;
# the HARPS-N days of ${SG_HARPSN_OUT} are in the CLV drift fit and in exposures.csv too:
#   bash supergranulation_haywood_neid_collect.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_supergranulation_haywood_neid.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

status=0
for variant in "${SG_VARIANTS[@]}"; do
    results=${variant%%:*}
    options=${variant#*:}
    echo "==== ${results}: ${options}"
    python "${PIPELINES}/supergranulation_haywood_neid.py" collect --out "$SG_OUT" --start "$SG_START" --conf "$SG_CONF" \
           --mps-folder "$SG_MPS" --mask "$SG_MASK" --n-rings "$SG_RINGS" $SG_QUALITY --results "$results" \
           $options || status=1
done
exit $status

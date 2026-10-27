#!/bin/bash
# Step 5 (one job, after all the simulations): table of the MAD / RMS of every set (sweep_stats.csv) and the figures,
# for every bisector (or only the bisectors given) and every part of TOTALS, in
# ${BASE_C}/results_${TAG}${RUN}/<bisector>_<part>/:
#   bash collect_components.sh [<bisector> ...]
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"

status=0
for b in ${*:-$BISECTOR_NAMES}; do
    for part in $TOTALS; do
        python -m sunsim.observations collect "${SETS}/${b}/${part}" --out "${BASE_C}/results_${TAG}${RUN}/${b}_${part}" \
            || status=1
    done
done
exit $status

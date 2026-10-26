#!/bin/bash
# SCRIPT PRE of sweep_components.dag and sweep_components_keepbis.dag (on the submit node, before the grids): the lists
# the submit files queue from, without the comments of the parameter and bisector files:
#   jobs_components_bisectors.txt : one bisector per line (build_grids_components.sub)
#   jobs_components.txt           : noise_th, I_th, A_th, Temp, bisector: every line of parameters_components.txt for
#                                   every bisector (run_components.sub)
# Every bisector of bisectors_components.txt, or only the bisectors given:
#   bash jobs_components.sh [<bisector> ...]
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"

bisectors=${*:-$BISECTOR_NAMES}
for b in $bisectors; do
    if ! [[ " $BISECTOR_NAMES " == *" $b "* ]]; then
        echo "ERROR: no bisector '$b' in $BISECTORS"
        exit 1
    fi
done
for b in $bisectors; do echo "$b"; done > "${PIPELINES}/jobs_components_bisectors.txt"
awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {print $1, $2, $3, $4}' "$PARAMS" | while read -r line; do
    for b in $bisectors; do echo "$line $b"; done
done > "${PIPELINES}/jobs_components.txt"
echo "$(wc -l < "${PIPELINES}/jobs_components_bisectors.txt") bisectors," \
     "$(wc -l < "${PIPELINES}/jobs_components.txt") simulation jobs"
[ -s "${PIPELINES}/jobs_components.txt" ]

#!/bin/bash
# SCRIPT PRE of sweep_components.dag (on the submit node, before the grids): the lists the submit files queue from,
# without the comments of the parameter and bisector files:
#   jobs_components_bisectors.txt : one bisector per line (build_grids_components.sub)
#   jobs_components.txt           : noise_th, I_th, A_th, Temp, bisector: every line of parameters_components.txt for
#                                   every bisector (run_components.sub)
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"

for b in $BISECTOR_NAMES; do echo "$b"; done > "${PIPELINES}/jobs_components_bisectors.txt"
awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {print $1, $2, $3, $4}' "$PARAMS" | while read -r line; do
    for b in $BISECTOR_NAMES; do echo "$line $b"; done
done > "${PIPELINES}/jobs_components.txt"
echo "$(wc -l < "${PIPELINES}/jobs_components_bisectors.txt") bisectors," \
     "$(wc -l < "${PIPELINES}/jobs_components.txt") simulation jobs"
[ -s "${PIPELINES}/jobs_components.txt" ]

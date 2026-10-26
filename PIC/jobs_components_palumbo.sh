#!/bin/bash
# SCRIPT PRE of sweep_components_palumbo.dag: the lists of jobs_components.sh for the bisector palumbo of
# bisectors_components.txt only:
#   jobs_components_palumbo_bisectors.txt : palumbo (build_grids_components_palumbo.sub)
#   jobs_components_palumbo.txt           : noise_th, I_th, A_th, Temp, palumbo: every line of parameters_components.txt
#                                           (run_components_palumbo.sub)
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"
b=palumbo

if [[ " $BISECTOR_NAMES " != *" $b "* ]]; then
    echo "ERROR: no bisector '$b' in $BISECTORS"
    exit 1
fi
echo "$b" > "${PIPELINES}/jobs_components_palumbo_bisectors.txt"
awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {print $1, $2, $3, $4}' "$PARAMS" | while read -r line; do
    echo "$line $b"
done > "${PIPELINES}/jobs_components_palumbo.txt"
echo "$(wc -l < "${PIPELINES}/jobs_components_palumbo.txt") simulation jobs"
[ -s "${PIPELINES}/jobs_components_palumbo.txt" ]

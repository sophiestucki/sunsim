#!/bin/bash
# Step 1 (one job): the configuration (60 rings) and the grids of every Temp of the parameter file, computed once
# (the same files as the 4096 px sweep: nothing is computed for a Temp whose grids are already there):
#   bash build_grids_fraction.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_fraction.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

for Temp in $(awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {printf "%.0f\n", $4}' "$PARAMS" | sort -u); do
    conf="${BASE}/confs/sunsim_T_${Temp}_N${N_RINGS}.conf"
    write_conf "$Temp" "$conf"
    python -m sunsim.sdo --build-grids --conf "$conf" --grids-cache "${BASE}/grids/grids_T_${Temp}_N${N_RINGS}.pkl" || exit 1
done

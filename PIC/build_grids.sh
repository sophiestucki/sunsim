#!/bin/bash
# Step 1 (one job): the configuration and the grids of every Temp of the parameter file, computed once:
#   bash build_grids.sh <threads> <parameter file>      (lines: noise_th, I_th, A_th, Temp)
PIPELINES=/data/spotless/scratch/sstucki/starsim/pipelines
source "${PIPELINES}/settings.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1
params=$2

for Temp in $(awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {printf "%.0f\n", $4}' "$params" | sort -u); do
    conf="${BASE}/confs/sunsim_T_${Temp}.conf"
    write_conf "$Temp" "$conf"
    python -m sunsim.sdo --build-grids --conf "$conf" --grids-cache "${BASE}/grids/grids_T_${Temp}.pkl" || exit 1
done

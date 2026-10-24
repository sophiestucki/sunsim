#!/bin/bash
# Step 1 (one job): the configuration (N_RINGS rings, Dumusque bisector times mu) and the grids of every Temp of the two
# parameter files, computed once (nothing is computed for a Temp whose grids are already there):
#   bash build_grids_720_dumusque_mu.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_720_dumusque_mu.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

for Temp in $(for T in $TEMPS; do printf "%.0f\n" "$T"; done | sort -u); do
    conf="${CONFS}/sunsim_T_${Temp}_N${N_RINGS}.conf"
    write_conf_dumusque_mu "$Temp" "$conf" || exit 1
    python -m sunsim.sdo --build-grids --conf "$conf" --grids-cache "${GRIDS}/grids_T_${Temp}_N${N_RINGS}.pkl" || exit 1
done

#!/bin/bash
# Step 1 of sweep_network.dag (one job): the configuration (N_RINGS rings, ccf_norm = 2, Dumusque bisector times mu) and
# the grids of every Temp of parameters_network.txt (nothing is computed for a Temp whose grids are already there):
#   bash build_grids_network.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_network.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

for Temp in $(for T in $TEMPS; do printf "%.0f\n" "$T"; done | sort -u); do
    conf="${CONFS}/sunsim_T_${Temp}_N${N_RINGS}.conf"
    write_conf_network "$Temp" "$conf" || exit 1
    python -m sunsim.sdo --build-grids --conf "$conf" --grids-cache "${GRIDS}/grids_T_${Temp}_N${N_RINGS}.pkl" || exit 1
done

#!/bin/bash
# Step 1 of sweep_noise.dag (one job): the configuration (N_RINGS rings, ccf_norm = 2, Dumusque bisector times mu) and
# the grids of every Temp of the two parameter files, computed once (nothing is computed for a Temp whose grids are
# already there):
#   bash build_grids_noise.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_noise.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

for Temp in $(for T in $TEMPS; do printf "%.0f\n" "$T"; done | sort -u); do
    conf="${CONFS}/sunsim_T_${Temp}_N${N_RINGS}.conf"
    write_conf_noise "$Temp" "$conf" || exit 1
    python -m sunsim.sdo --build-grids --conf "$conf" --grids-cache "${GRIDS}/grids_T_${Temp}_N${N_RINGS}.pkl" || exit 1
done

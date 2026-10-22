#!/bin/bash
# Step 1 (one job): the configuration (N_RINGS rings) and the grids of every Temp of the two parameter files, computed
# once (nothing is computed for a Temp whose grids are already there; the grids do not depend on the magnetograms):
#   bash build_grids_720.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_720.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

for Temp in $(for T in $TEMPS; do printf "%.0f\n" "$T"; done | sort -u); do
    conf="${BASE}/confs/sunsim_T_${Temp}_N${N_RINGS}.conf"
    write_conf "$Temp" "$conf"
    python -m sunsim.sdo --build-grids --conf "$conf" --grids-cache "${BASE}/grids/grids_T_${Temp}_N${N_RINGS}.pkl" || exit 1
done

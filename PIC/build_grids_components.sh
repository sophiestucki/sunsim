#!/bin/bash
# Step 1 (one job per bisector): the configuration (N_RINGS rings, spot bisector <bisector> of bisectors_components.txt)
# and the grids of every Temp of parameters_components.txt (nothing is computed for a Temp whose grids are there):
#   bash build_grids_components.sh <threads> <bisector>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1
bisector=$2

for Temp in $(for T in $TEMPS; do printf "%.0f\n" "$T"; done | sort -u); do
    conf="${CONFS}/sunsim_T_${Temp}_N${N_RINGS}_${bisector}.conf"
    write_conf_bisector "$bisector" "$Temp" "$conf" || exit 1
    python -m sunsim.sdo --build-grids --conf "$conf" \
           --grids-cache "${GRIDS}/grids_T_${Temp}_N${N_RINGS}_${bisector}.pkl" || exit 1
done

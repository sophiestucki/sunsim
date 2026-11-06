#!/bin/bash
# Step 1 of sweep_facula_B.dag (one job per line of parameters_facula_B.txt): the configuration (N_RINGS rings,
# ccf_norm = 2, Dumusque bisector times mu, facula_B = B_fc) and its grids (nothing is computed if they are there):
#   bash build_grids_facula_B.sh <threads> <Temp> <B_fc>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_facula_B.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

Temp=$(printf "%.0f" "$2")
cname=$(conf_name_facula_B "$Temp" "$3")
conf="${CONFS}/sunsim_${cname}.conf"
write_conf_facula_B "$Temp" "$3" "$conf" || exit 1
python -m sunsim.sdo --build-grids --conf "$conf" --grids-cache "${GRIDS}/grids_${cname}.pkl"

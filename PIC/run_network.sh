#!/bin/bash
# Step 4 of sweep_network.dag (one job per line of parameters_network.txt): simulation of the decade for every part
# (PARTS of settings_network.sh, the filling factors of aggregate_network.sh; the grids of build_grids_network.sh):
# light curve, RV, FWHM, BIS, contrast and CCF time series in ${SETS}/<part>/<name>/, then the MAD and RMS against the
# observations (stats.json) of the TOTALS:
#   bash run_network.sh <threads> <noise_th> <B_th> <I_th> <A_th> <min_px> <B_sat> <Temp>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_network.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

noise_th=$2
B_th=$3
I_th=$4
A_th=$5
min_px=$6
B_sat=$7
Temp=$(printf "%.0f" "$8")

name="noise_th_$(printf "%.1f" "$noise_th")_B_th_$(printf "%.1f" "$B_th")_I_th_$(printf "%.2f" "$I_th")_A_th_$(printf "%.1f" "$A_th")_min_px_$(printf "%.0f" "$min_px")_B_sat_$(printf "%.1f" "$B_sat")_T_${Temp}_${TAG}"
conf="${CONFS}/sunsim_T_${Temp}_N${N_RINGS}.conf"
grids="${GRIDS}/grids_T_${Temp}_N${N_RINGS}.pkl"
label=$(set_label_network "$noise_th" "$B_th" "$I_th" "$A_th" "$min_px" "$B_sat")

folders=()
for part in $PARTS; do
    src="${PARTS_DIR}/${label}/${part}"
    for f in "$grids" "${src}/filling_factors_N${N_RINGS}.npz"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: $f is missing: run build_grids_network.sh, identify_network.sh and aggregate_network.sh first"
            exit 1
        fi
    done
    folder="${SETS}/${part}/${name}"
    mkdir -p "$folder"
    for f in "filling_factors_N${N_RINGS}.npz" summary.csv parameters.txt; do
        ln -sfn "${src}/${f}" "${folder}/${f}"
    done
    folders+=("$folder")
done

python -c "
import sys
from sunsim.sdo import simulate_sdo_sets
simulate_sdo_sets(sys.argv[1], sys.argv[3:], grids_cache=sys.argv[2], chunk_size=500)
" "$conf" "$grids" "${folders[@]}"
status=$?
if [ $status -ne 0 ]; then
    echo "Simulation failed (outputs so far kept in ${SETS}/<part>/${name})"
    exit $status
fi

# MAD and RMS of the totals against HARPS-N (RV, FWHM, BIS) and SORCE / TSIS-1 (TSI): stats.json
for part in $TOTALS; do
    python -m sunsim.observations stats "${SETS}/${part}/${name}" --start "$START" --harpsn "$HARPSN" \
                                        --interventions "$INTERVENTIONS" --sorce "$SORCE" --tsis "$TSIS" $WINDOW \
                                        --param noise_th="$noise_th" B_th="$B_th" I_th="$I_th" A_th="$A_th" \
                                                min_px="$min_px" B_sat="$B_sat" T_sp="$Temp" || exit 1
done
echo "Done: ${SETS}/<part>/${name}"

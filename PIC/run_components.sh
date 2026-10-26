#!/bin/bash
# Step 4 (one job per line of parameters_components.txt and bisector): simulation of the decade for every part of the
# set (PARTS of settings_components.sh, the filling factors of aggregate_components.sh; the grids of
# build_grids_components.sh: spot bisector <bisector>, computed once for the parts): light curve, RV, FWHM, BIS,
# contrast and CCF time series in ${SETS}/<bisector>/<part>/<name>/, then the MAD and RMS against the observations
# (stats.json) of the TOTALS:
#   bash run_components.sh <threads> <noise_th> <I_th> <A_th> <Temp> <bisector>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_components.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

noise_th=$2
I_th=$3
A_th=$4
Temp=$(printf "%.0f" "$5")
bisector=$6

name="noise_th_$(printf "%.1f" "$noise_th")_I_th_$(printf "%.2f" "$I_th")_A_th_$(printf "%.1f" "$A_th")_T_${Temp}_${TAG}"
conf="${CONFS}/sunsim_T_${Temp}_N${N_RINGS}_${bisector}.conf"
grids="${GRIDS}/grids_T_${Temp}_N${N_RINGS}_${bisector}.pkl"
label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise_th, I_th=$I_th, A_th=$A_th, faculae='plage')))")

folders=()
for part in $PARTS; do
    src="${PARTS_DIR}/${label}/${part}"
    for f in "$grids" "${src}/filling_factors_N${N_RINGS}.npz"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: $f is missing: run build_grids_components.sh, identify_components.sh and" \
                 "aggregate_components.sh first"
            exit 1
        fi
    done
    folder="${SETS}/${bisector}/${part}/${name}"
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
    echo "Simulation failed (outputs so far kept in ${SETS}/${bisector}/<part>/${name})"
    exit $status
fi

# MAD and RMS of the totals against HARPS-N (RV, FWHM, BIS) and SORCE / TSIS-1 (TSI): stats.json
for part in $TOTALS; do
    python -m sunsim.observations stats "${SETS}/${bisector}/${part}/${name}" --start "$START" --harpsn "$HARPSN" \
                                        --interventions "$INTERVENTIONS" --sorce "$SORCE" --tsis "$TSIS" $WINDOW \
                                        --param noise_th="$noise_th" I_th="$I_th" A_th="$A_th" T_sp="$Temp" || exit 1
done

# the time series of every part also in one folder per bisector (time [d] in the first column)
out="${BASE_C}/opt_rv_${TAG}/${bisector}"
mkdir -p "$out"
for part in $PARTS; do
    for q in rv lc fwhm bis contrast ccf ff; do
        cp "${SETS}/${bisector}/${part}/${name}/${q}.txt" "${out}/${q}_${name}_${part}.txt"
    done
done
echo "Done: ${SETS}/${bisector}/<part>/${name}"

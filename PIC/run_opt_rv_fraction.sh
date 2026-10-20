#!/bin/bash
# Step 4 (one job per line of the parameter file): simulation of the decade from the identification of its thresholds
# (4096 px, 60 rings, facular fractions; the grids of build_grids_fraction.sh): light curve, RV, FWHM, BIS, contrast and
# CCF time series, then the MAD and RMS against the observations (stats.json):
#   bash run_opt_rv_fraction.sh <threads> <noise_th> <I_th> <B_sat> <Temp>
# The folder of the line links the filling factors of its set (shared by all the Temp of the set).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_fraction.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

noise_th=$2
I_th=$3
B_sat=$4
Temp=$(printf "%.0f" "$5")

name="noise_th_$(printf "%.1f" "$noise_th")_I_th_$(printf "%.2f" "$I_th")_B_sat_$(printf "%.1f" "$B_sat")_T_${Temp}_${TAG}"
folder="${SETS}/${name}"
conf="${BASE}/confs/sunsim_T_${Temp}_N${N_RINGS}.conf"
grids="${BASE}/grids/grids_T_${Temp}_N${N_RINGS}.pkl"
label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise_th, I_th=$I_th, B_sat=$B_sat, faculae='fraction')))")
ident="${IDENT}/${label}"

for f in "$grids" "${ident}/filling_factors_N${N_RINGS}.npz"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f is missing: run build_grids_fraction.sh, identify_fraction.sh and aggregate_fraction.sh first"
        exit 1
    fi
done
mkdir -p "$folder"
for f in "filling_factors_N${N_RINGS}.npz" summary.csv parameters.txt; do
    ln -sfn "${ident}/${f}" "${folder}/${f}"
done

python -m sunsim.sdo --simulate-only --out "$folder" --conf "$conf" --grids-cache "$grids" --chunk-size 500
status=$?
if [ $status -ne 0 ]; then
    echo "Simulation failed (outputs so far kept in $folder)"
    exit $status
fi

# MAD and RMS of this set against HARPS-N (RV, FWHM, BIS) and SORCE / TSIS-1 (TSI): ${folder}/stats.json
python -m sunsim.observations stats "$folder" --start "$START" --harpsn "$HARPSN" --interventions "$INTERVENTIONS" \
                                    --sorce "$SORCE" --tsis "$TSIS" $WINDOW \
                                    --param noise_th="$noise_th" I_th="$I_th" B_sat="$B_sat" T_sp="$Temp" || exit 1

# the time series also in one folder (time [d] in the first column)
mkdir -p "${BASE}/opt_rv_${TAG}"
for q in rv lc fwhm bis contrast ccf; do
    cp "${folder}/${q}.txt" "${BASE}/opt_rv_${TAG}/${q}_${name}.txt"
done
echo "Done: ${folder}"

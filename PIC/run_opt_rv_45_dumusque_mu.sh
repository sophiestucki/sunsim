#!/bin/bash
# Step 2 (one job per line of the parameter file of KIND): simulation of the decade from the identification of its
# thresholds (4096 px, 60 rings, 45 s magnetograms; the identification of sweep_4096.dag or sweep_fraction.dag, the
# grids of build_grids_45_dumusque_mu.sh: Dumusque bisector times mu): light curve, RV, FWHM, BIS, contrast and CCF
# time series, then the MAD and RMS against the observations (stats.json):
#   bash run_opt_rv_45_dumusque_mu.sh <plage|fraction> <threads> <noise_th> <I_th> <A_th|B_sat> <Temp>
# The folder of the line links the filling factors of its set (shared by all the Temp of the set).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
KIND=$1
source "${PIPELINES}/settings_45_dumusque_mu.sh"
export OMP_NUM_THREADS=$2 NUMBA_NUM_THREADS=$2

noise_th=$3
I_th=$4
X=$5
Temp=$(printf "%.0f" "$6")

name="noise_th_$(printf "%.1f" "$noise_th")_I_th_$(printf "%.2f" "$I_th")_${X_NAME}_$(printf "%.1f" "$X")_T_${Temp}_${TAG}"
folder="${SETS}/${name}"
conf="${CONFS}/sunsim_T_${Temp}_N${N_RINGS}.conf"
grids="${GRIDS}/grids_T_${Temp}_N${N_RINGS}.pkl"
label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise_th, I_th=$I_th, $X_NAME=$X, faculae='$FACULAE')))")
ident="${IDENT}/${label}"

for f in "$grids" "${ident}/filling_factors_N${N_RINGS}.npz"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f is missing: run sweep_4096.dag or sweep_fraction.dag (up to aggregate) and" \
             "build_grids_45_dumusque_mu.sh first"
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
                                    --param noise_th="$noise_th" I_th="$I_th" "$X_NAME"="$X" T_sp="$Temp" || exit 1

# the time series also in one folder (time [d] in the first column)
mkdir -p "${BASE_DMU}/opt_rv_${TAG}"
for q in rv lc fwhm bis contrast ccf; do
    cp "${folder}/${q}.txt" "${BASE_DMU}/opt_rv_${TAG}/${q}_${name}.txt"
done
echo "Done: ${folder}"

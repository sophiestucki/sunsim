#!/bin/bash
# Step 4 of sweep_facula_B.dag (one job per line of parameters_facula_B.txt): simulation of the decade from the
# identification of its thresholds with the grids of its (Temp, B_fc) (build_grids_facula_B.sh): light curve, RV,
# FWHM, BIS, contrast and CCF time series, then the MAD and RMS against the observations (stats.json):
#   bash run_facula_B.sh <threads> <noise_th> <B_th> <I_th> <A_th> <B_fc> <Temp>
# The folder of the line links the filling factors of its set (shared by all the B_fc).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_facula_B.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

noise_th=$2
B_th=$3
I_th=$4
A_th=$5
B_fc=$6
Temp=$(printf "%.0f" "$7")

name="noise_th_$(printf "%.1f" "$noise_th")_B_th_$(printf "%.1f" "$B_th")_I_th_$(printf "%.2f" "$I_th")_A_th_$(printf "%.1f" "$A_th")_B_fc_$(printf "%.1f" "$B_fc")_T_${Temp}_${TAG}"
folder="${SETS}/${name}"
cname=$(conf_name_facula_B "$Temp" "$B_fc")
conf="${CONFS}/sunsim_${cname}.conf"
grids="${GRIDS}/grids_${cname}.pkl"
label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise_th, B_th=$B_th, I_th=$I_th, A_th=$A_th, faculae='$FACULAE')))")
ident="${IDENT}/${label}"

for f in "$conf" "$grids" "${ident}/filling_factors_N${N_RINGS}.npz"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f is missing: run build_grids_facula_B.sh, identify_facula_B.sh and aggregate_facula_B.sh first"
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
                                    --param noise_th="$noise_th" B_th="$B_th" I_th="$I_th" A_th="$A_th" \
                                            B_fc="$B_fc" T_sp="$Temp" || exit 1

# the time series also in one folder (time [d] in the first column)
mkdir -p "${BASE_FB}/opt_rv_${TAG}"
for q in rv lc fwhm bis contrast ccf; do
    cp "${folder}/${q}.txt" "${BASE_FB}/opt_rv_${TAG}/${q}_${name}.txt"
done
echo "Done: ${folder}"

#!/bin/bash
# Step 2 (one job per set of thresholds): identification of every day with SunSim (sunsim.sdo, no maps written: only
# the filling factors of the cells), simulation with the grids of build_grids.sh (read, not computed): light curve,
# RV, FWHM, BIS, contrast and CCF time series, then the MAD and RMS against the observations (stats.json):
#   bash run_10_opt_rv.sh <workers> <noise_th> <I_th> <A_th> <Temp>
# e.g. bash run_10_opt_rv.sh 8 8 0.95 20 300
# <workers>: days identified at the same time (processes, one CPU each); the rest is serial (one thread)
# A job that stops can be started again: the days and the simulation already done are skipped.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

noise_th=$2
I_th=$3
A_th=$4
Temp=$(printf "%.0f" "$5")

name="noise_th_$(printf "%.1f" "$noise_th")_I_th_$(printf "%.2f" "$I_th")_A_th_$(printf "%.1f" "$A_th")_T_${Temp}_size_${SIZE}"
folder="${BASE}/sets/${name}"
conf="${BASE}/confs/sunsim_T_${Temp}.conf"
grids="${BASE}/grids/grids_T_${Temp}.pkl"

if [ ! -f "$grids" ]; then
    echo "ERROR: no grids for T = ${Temp} (${grids}): run build_grids.sh first"
    exit 1
fi

python -m sunsim.sdo --data "$SDO_DATA" --out "$folder" --start "$START" --end "$END" --size "$SIZE" \
                   --noise-th "$noise_th" --I-th "$I_th" --A-th "$A_th" --format none \
                   --conf "$conf" --grids-cache "$grids" --workers "$1" --chunk-size 500
status=$?
if [ $status -ne 0 ]; then
    echo "Simulation failed (outputs so far kept in $folder)"
    exit $status
fi

# MAD and RMS of this set against HARPS-N (RV, FWHM, BIS) and SORCE / TSIS-1 (TSI): ${folder}/stats.json
python -m sunsim.observations stats "$folder" --start "$START" --harpsn "$HARPSN" --interventions "$INTERVENTIONS" \
                                    --sorce "$SORCE" --tsis "$TSIS" $WINDOW \
                                    --param noise_th="$noise_th" I_th="$I_th" A_th="$A_th" T_sp="$Temp" || exit 1

# the time series also in one folder, with the names of the previous runs (time [d] in the first column)
mkdir -p "${BASE}/opt_rv"
for q in rv lc fwhm bis contrast ccf; do
    cp "${folder}/${q}.txt" "${BASE}/opt_rv/${q}_${name}.txt"
done
echo "Done: ${folder}"

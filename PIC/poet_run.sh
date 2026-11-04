#!/bin/bash
# Step 3 (one job per line of poet_jobs.txt): simulation of the epochs of one set of thresholds with one instrument
# (None: the CCF of the whole spectrum; PoET: R = 190000, CCFs of the 170 orders with the normalised blaze) and one
# CCF mask (ESPRESSO_SUN, PoET_noflag), then its values at every exposure (exposures_simulated.csv):
#   bash poet_run.sh <threads> <noise_th> <I_th> <A_th> <Temp> <instrument> <mask>
# The grids of the configuration are computed in the job (${POET_OUT}/grids/, read again if the job is run again).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_poet.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

noise_th=$2
I_th=$3
A_th=$4
Temp=$(printf "%.0f" "$5")
instrument=$6
mask=$7

tag="T_${Temp}_N${N_RINGS}_${instrument}_${mask}"
name="noise_th_$(printf "%.1f" "$noise_th")_I_th_$(printf "%.2f" "$I_th")_A_th_$(printf "%.1f" "$A_th")_${tag}"
folder="${POET_OUT}/sets/${name}"
conf="${POET_OUT}/confs/sunsim_${tag}.conf"
grids="${POET_OUT}/grids/grids_${tag}.pkl"
label=$(python -c "from sunsim.sdo import set_label; print(set_label(dict(noise_th=$noise_th, I_th=$I_th, A_th=$A_th)))")
ident="${POET_OUT}/ident/${label}"

if [ ! -f "${ident}/filling_factors_N${N_RINGS}.npz" ]; then
    echo "ERROR: ${ident}/filling_factors_N${N_RINGS}.npz is missing: run poet_day.sh and poet_collect.sh first"
    exit 1
fi
write_conf_poet "$Temp" "$instrument" "$mask" "$conf" || exit 1
mkdir -p "$folder" "${POET_OUT}/grids"
for f in "filling_factors_N${N_RINGS}.npz" summary.csv parameters.txt; do
    ln -sfn "${ident}/${f}" "${folder}/${f}"
done
cat > "${folder}/poet_run.json" << JSON
{"set": "${label}", "noise_th": ${noise_th}, "I_th": ${I_th}, "A_th": ${A_th}, "T_sp": ${Temp},
 "instrument": "${instrument}", "mask": "${mask}"}
JSON

python -m sunsim.sdo --simulate-only --out "$folder" --conf "$conf" --grids-cache "$grids" --chunk-size 200
status=$?
if [ $status -ne 0 ]; then
    echo "Simulation failed (outputs so far kept in $folder)"
    exit $status
fi
python "${PIPELINES}/poet_epochs.py" exposures --out "$POET_OUT" --set "$folder" --n-rings "$N_RINGS" || exit 1
echo "Done: ${folder}"

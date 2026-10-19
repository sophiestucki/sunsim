#!/bin/bash
# Step 2 (one job, after all the days): calibration of the HMI velocities over the whole period, simulation of every
# exposure (maps; maps + dopplergram; maps + dopplergram + p modes), statistics and figures of every day, of the
# daily averages over the period (maps against maps + dopplergram) and of the full dataset in ${SG_OUT}/results:
#   bash supergranulation_collect.sh <threads>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_supergranulation.sh"
export OMP_NUM_THREADS=$1 NUMBA_NUM_THREADS=$1

python "${PIPELINES}/supergranulation.py" collect --out "$SG_OUT" --start "$SG_START" --conf "$SG_CONF" \
       --mps-folder "$SG_MPS" --mask "$SG_MASK" --interventions "$INTERVENTIONS" --n-rings "$SG_RINGS"

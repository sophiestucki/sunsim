#!/bin/bash
# Step 4 (one job): all the simulations at every exposure in ${POET_OUT}/results/poet_simulations.csv (one row per
# exposure and simulation) and the standard deviation of every observable (poet_simulations_std.csv).
#   bash poet_table.sh
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_poet.sh"

python "${PIPELINES}/poet_epochs.py" table --out "$POET_OUT"

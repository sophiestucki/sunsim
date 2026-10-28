#!/bin/bash
# Step 2 of uninformed_decade_dsat.dag (one job, after the training): as uninformed_synth.sh with the model of
# ${BASE_D}/model: ${BASE_D}/synthetic  (bash uninformed_synth_dsat.sh <workers>)
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_uninformed_dsat.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/uninformed_decade.py" synth --out "$BASE_D" --maps "$MAPS_U" --min "$SYN_MIN" --max "$SYN_MAX" \
       --workers "$1"

#!/bin/bash
# Step 2 (one job, after all the maps): features of every day, statistics, fit of the generator, closed loop and
# calibrations: ${BASE_U}/model/model.json  (bash uninformed_train.sh <workers>)
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_uninformed.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/uninformed_decade.py" train --out "$BASE_U" --start "$START" --workers "$1"

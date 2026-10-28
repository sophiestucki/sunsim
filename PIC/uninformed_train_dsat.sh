#!/bin/bash
# Step 1 of uninformed_decade_dsat.dag (one job): the training of uninformed_train.sh with the decay rate constant above
# A_SAT, on the masks of ${BASE_U}/maps: ${BASE_D}/model/model.json  (bash uninformed_train_dsat.sh <workers>)
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_uninformed_dsat.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

# the features of the SDO days do not depend on the model: those of uninformed_decade.dag, not computed again
mkdir -p "${BASE_D}/model"
if [ ! -f "${BASE_D}/model/features.pkl" ] && [ -f "${BASE_U}/model/features.pkl" ]; then
    cp "${BASE_U}/model/features.pkl" "${BASE_D}/model/features.pkl"
fi

python "${PIPELINES}/uninformed_decade.py" train --out "$BASE_D" --maps "$MAPS_U" --start "$START" \
       --decay-sat "$A_SAT" --workers "$1"

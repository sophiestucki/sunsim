#!/bin/bash
# Step 3 of sweep_network.dag (one job, after all the blocks): the days still missing, then the filling factors of the
# parts of every set in ${PARTS_DIR}/<set label>/<part>/ and the histograms of the noise diagnostics
# (${IDENT}/diag/<key label>/histograms.npz):
#   bash aggregate_network.sh <workers>
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_network.sh"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "${PIPELINES}/identify_network.py" --data "$SDO_DATA" --out "$IDENT" --start "$START" --end "$END" \
       --n-rings "$N_RINGS" --params "$PARAMS" --diag-noise-th "$DIAG_NOISE_TH" --diag-B-th $DIAG_B_TH \
       --diag-I-th "$DIAG_I_TH" --diag-B-sat "$DIAG_B_SAT" --magnetogram "$MAGNETOGRAM" \
       --block 0 --n-blocks 1 --workers "$1" --aggregate --parts "$PARTS_DIR"

#!/bin/bash
# Step 5 of sweep_facula_B.dag (one job, after all the simulations): table of the MAD / RMS of every set,
# ${BASE_FB}/results_${TAG}/sweep_stats.csv (the figures of the sweep: facula_B.ipynb)
#   bash collect_facula_B.sh
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_facula_B.sh"

out="${BASE_FB}/results_${TAG}"
mkdir -p "$out"
python -c "
import sys
from sunsim.observations import load_sweep_stats
df = load_sweep_stats(sys.argv[1])
df.to_csv(sys.argv[2], index=False)
print('%d sets -> %s' % (len(df), sys.argv[2]))
" "$SETS" "${out}/sweep_stats.csv"

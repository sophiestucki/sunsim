#!/bin/bash
# Step 5 of sweep_network.dag (one job, after all the simulations): table of the MAD / RMS of every set of every part
# of TOTALS, ${BASE_NW}/results_${TAG}/<part>/sweep_stats.csv (the figures of the sweep: network.ipynb)
#   bash collect_network.sh
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_network.sh"

status=0
for part in $TOTALS; do
    out="${BASE_NW}/results_${TAG}/${part}"
    mkdir -p "$out"
    python -c "
import sys
from sunsim.observations import load_sweep_stats
df = load_sweep_stats(sys.argv[1])
df.to_csv(sys.argv[2], index=False)
print('%d sets -> %s' % (len(df), sys.argv[2]))
" "${SETS}/${part}" "${out}/sweep_stats.csv" || status=1
done
exit $status

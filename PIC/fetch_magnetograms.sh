#!/bin/bash
# Step 0 (N_DOWNLOADS jobs): the magnetograms MAGNETOGRAM (720 s) of the same T_REC as the continuum images of
# SDO_DATA, one block of the days per job, from the JSOC storage (no email; sunsim.sdo.fetch_magnetograms), written next
# to them (~14 MB each, ~45 GB for the decade):  bash fetch_magnetograms.sh <block>
# A block can be started again: the days already there are skipped. Exit code 1 if a download failed.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_720.sh"

python -c "
import sys
from sunsim import sdo
written, failed = sdo.fetch_magnetograms('$SDO_DATA', '$START', '$END', magnetogram='$MAGNETOGRAM', block=$1,
                                         n_blocks=$N_DOWNLOADS)
print('block $1: %d downloaded, %d failed %s' % (len(written), len(failed), ' '.join(failed)))
sys.exit(1 if failed else 0)
"

#!/bin/bash
# Settings of the SunSim simulation at the times of the PoET exposures (poet.dag), on top of settings.sh (environment,
# BASE, write_conf). Sourced by poet_day.sh, poet_collect.sh, poet_run.sh and poet_table.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

POET_TIMES=${PIPELINES}/poet_jdb.csv           # column jdb: BJD (TDB) of the PoET exposures
POET_START=2026-04-06                          # t = 0 of the simulation (00:00 UTC): the first day of POET_TIMES
POET_OUT=${BASE}/poet                          # days/, ident/, confs/, grids/, sets/, results/ (poet_epochs.py)
PARAMS=${PIPELINES}/parameters_poet.txt        # lines: noise_th, I_th, A_th, Temp (one set of thresholds each)
SIZE=4096                                      # pixels of the HMI images (full resolution)
N_RINGS=60                                     # rings of the SunSim grid (write_conf uses it)
MAX_DT=720                                     # [s] exposures farther than this from a 720 s continuum are left out
MASKS=${PIPELINES}/masks                       # <mask>.mas of poet_masks.py: ESPRESSO_SUN, PoET_noflag

# write_conf of settings.sh with the instrument (None or PoET) and the CCF mask (a name of ${MASKS})
write_conf_poet() {
    local Temp=$1 instrument=$2 mask=$3 file=$4
    write_conf "$Temp" "$file"
    awk -v inst="$instrument" -v mask="${MASKS}/${mask}.mas" '
        /^instrument *:/ {print "instrument: " inst; next}
        /^ccf_mask *:/   {print "ccf_mask: " mask; next}
        {print}' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    if ! grep -q "^instrument: ${instrument}$" "$file" || ! grep -q "^ccf_mask: ${MASKS}/${mask}.mas$" "$file"; then
        echo "ERROR: no instrument or ccf_mask line in $file"
        return 1
    fi
}

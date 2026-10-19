#!/bin/bash
# Settings of the decade at full resolution (4096 px, 60 rings), on top of settings.sh (period, environment,
# observations, write_conf). Sourced by build_grids_4096.sh, identify_4096.sh, aggregate_4096.sh, run_opt_rv_4096.sh
# and collect_stats_4096.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

SIZE=4096                              # pixels of the HMI images (full resolution)
N_RINGS=60                             # rings of the SunSim grid (write_conf uses it)
TAG=size_${SIZE}_N${N_RINGS}
SETS=${BASE}/sets_${TAG}               # <name>/ : simulation and stats.json of every line of the parameter file
IDENT=${SETS}/ident                    # <set label>/ : identification of every set of thresholds (shared by all Temp)
N_BLOCKS=40                            # identification: the days of the period in N_BLOCKS jobs
PARAMS=${PIPELINES}/parameters.txt     # lines: noise_th, I_th, A_th, Temp

# the values of the thresholds of the parameter file (the identification does all their combinations)
values() { awk -F'[ ,\t]+' -v c="$1" 'NF >= 4 && $1 !~ /^#/ {print $c + 0}' "$PARAMS" | sort -g -u | tr '\n' ' '; }
NOISE_VALUES=$(values 1)
I_VALUES=$(values 2)
A_VALUES=$(values 3)

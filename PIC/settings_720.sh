#!/bin/bash
# Settings of the decade at full resolution (4096 px, 60 rings) with the 720 s magnetograms (hmi.M_720s: noise ~5 G at the disc centre and 6 G near
# the limb, against ~10 and 13 G for hmi.M_45s), on top of settings.sh (period, environment, observations, write_conf).
# Everything is written in ${BASE720}, apart from the runs with the 45 s magnetograms. Two kinds of faculae, KIND set
# before sourcing (the scripts take it as first argument):
#   KIND=plage    : the groups of bright pixels larger than A_th (parameters_720_plage.txt: noise_th, I_th, A_th, Temp)
#   KIND=fraction : every bright pixel is min(|B|/mu/B_sat, 1) facula, no area threshold
#                   (parameters_720_fraction.txt: noise_th, I_th, B_sat, Temp)
# Sourced by fetch_magnetograms.sh (no KIND), build_grids_720.sh (no KIND), identify_720.sh, aggregate_720.sh,
# run_opt_rv_720.sh and collect_stats_720.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

MAGNETOGRAM=720s                       # in $SDO_DATA next to the continuum images (fetch_magnetograms.sh)
SIZE=4096                              # pixels of the HMI images (full resolution)
N_RINGS=60                             # rings of the SunSim grid (write_conf uses it; the grids of the 4096 px sweeps)
BASE720=${BASE}/m720s                  # sets_<TAG>/, opt_rv_<TAG>/, results_<TAG>/ of both kinds
N_DOWNLOADS=30                         # fetch_magnetograms: the days in N_DOWNLOADS jobs (= Queue of fetch_magnetograms.sub)
N_BLOCKS=80                            # identification: the days in N_BLOCKS jobs (= Queue of identify_720.sub)
N_AGGREGATE=40                         # aggregation: the sets in N_AGGREGATE jobs (= Queue of aggregate_720.sub)

# the values of the thresholds of a parameter file (the identification does all their combinations)
values() { awk -F'[ ,\t]+' -v c="$2" 'NF >= 4 && $1 !~ /^#/ {print $c + 0}' "$1" | sort -g -u | tr '\n' ' '; }
TEMPS=$(values "${PIPELINES}/parameters_720_plage.txt" 4; values "${PIPELINES}/parameters_720_fraction.txt" 4)

if [ -n "$KIND" ]; then
    case "$KIND" in
        plage)    X_NAME=A_th;  X_OPT=--A-th;  FACULAE=plage ;;
        fraction) X_NAME=B_sat; X_OPT=--B-sat; FACULAE=fraction ;;
        *) echo "ERROR: KIND must be plage or fraction, got '$KIND'"; exit 1 ;;
    esac
    TAG=${KIND}_m${MAGNETOGRAM}_size_${SIZE}_N${N_RINGS}
    SETS=${BASE720}/sets_${TAG}        # <name>/ : simulation and stats.json of every line of the parameter file
    IDENT=${SETS}/ident                # <set label>/ : identification of every set of thresholds (shared by all Temp)
    PARAMS=${PIPELINES}/parameters_720_${KIND}.txt
    NOISE_VALUES=$(values "$PARAMS" 1)
    I_VALUES=$(values "$PARAMS" 2)
    X_VALUES=$(values "$PARAMS" 3)
fi

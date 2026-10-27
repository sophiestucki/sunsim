#!/bin/bash
# Settings of sweep_components.dag: the decade at 4096 px (60 rings) with the area threshold method (plage = the groups
# of bright pixels larger than A_th), the time series of its parts simulated one by one, for several definitions of
# the spot bisector. On top of settings.sh (period, environment, observations, write_conf); everything in ${BASE_C}.
#   parameters_components.txt : noise_th, I_th, A_th, Temp (one simulation of every part per line and bisector)
#   bisectors_components.txt  : name, remove_bis_spot, convective_shift, dumusque_mu_scaling ('c0,c1,...': the
#                               Dumusque bisector of a ring times c0 + c1 mu + ...; '-': the same at every mu), one grid
#                               per bisector and Temp
# The parts (PARTS), from the identification of every set with faculae='plage' and faculae='plage+network'
# (split_components.py):
#   spot               : the spots alone (no facula)
#   plage              : the plage alone (no spot)
#   network            : the network alone (no spot), with the spectra of the faculae
#   spot_plage         : spots and plage, the network quiet Sun (the area threshold method of the other sweeps)
#   spot_plage_network : spots, plage and network
# Sourced by jobs_components.sh, build_grids_components.sh, identify_components.sh, aggregate_components.sh,
# run_components.sh and collect_components.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

MAGNETOGRAM=45s                        # hmi.M_45s; or 720s (next to the continuum images: fetch_magnetograms.sh)
SIZE=4096                              # pixels of the HMI images (full resolution)
N_RINGS=60                             # rings of the SunSim grid (write_conf uses it)
N_BLOCKS=80                            # identification: the days in N_BLOCKS jobs (= Queue of identify_components.sub)
TAG=components_m${MAGNETOGRAM}_size_${SIZE}_N${N_RINGS}
# ccf_norm of the configurations: 2 = raw ring CCFs (mps_highres: the spots dark, the faculae bright and the quiet disc
# limb darkened in the RV too); 0 = -ccf/min(ccf) + 1, every ring normalised (the runs before 2026-10-27). The
# simulations with ccf_norm = 2 go to folders with the suffix _rawccf; the identification and the filling factors of
# the parts are shared (those of sets_<TAG>)
CCF_NORM=2
RUN=$([ "$CCF_NORM" = 2 ] && echo _rawccf)
BASE_C=${BASE}/components              # confs/, grids/, sets_<TAG>/, opt_rv_<TAG>/, results_<TAG>/
CONFS=${BASE_C}/confs${RUN}
GRIDS=${BASE_C}/grids${RUN}
SETS=${BASE_C}/sets_${TAG}${RUN}       # <bisector>/<part>/<name>/ : simulation (and stats.json of the TOTALS)
IDENT=${BASE_C}/sets_${TAG}/ident      # <set label>/ and <set label>_network/ : identification (plage, plage+network)
PARTS_DIR=${BASE_C}/sets_${TAG}/parts  # <set label>/<part>/ : filling factors of every part (split_components.py)
PARAMS=${PIPELINES}/parameters_components.txt
BISECTORS=${PIPELINES}/bisectors_components.txt
PARTS="spot plage network spot_plage spot_plage_network"
TOTALS="spot_plage spot_plage_network"           # the parts compared with the observations (stats.json, collect)

# the values of a column of a file (the identification does all the combinations of the thresholds)
values() { awk -F'[ ,\t]+' -v c="$2" 'NF >= 4 && $1 !~ /^#/ {print $c + 0}' "$1" | sort -g -u | tr '\n' ' '; }
NOISE_VALUES=$(values "$PARAMS" 1)
I_VALUES=$(values "$PARAMS" 2)
A_VALUES=$(values "$PARAMS" 3)
TEMPS=$(values "$PARAMS" 4)
BISECTOR_NAMES=$(awk 'NF >= 4 && $1 !~ /^#/ {print $1}' "$BISECTORS" | tr '\n' ' ')

# write_conf of settings.sh with the spot bisector <bisector> of BISECTORS:  write_conf_bisector <bisector> <Temp> <file>
write_conf_bisector() {
    local def remove cs mu
    def=$(awk -v b="$1" 'NF >= 4 && $1 !~ /^#/ && $1 == b {print $2, $3, $4}' "$BISECTORS")
    if [ -z "$def" ]; then
        echo "ERROR: no bisector '$1' in $BISECTORS"
        return 1
    fi
    read -r remove cs mu <<< "$def"
    [ "$mu" = "-" ] && mu=""
    write_conf "$2" "$3"
    awk -v r="$remove" -v cs="$cs" -v mu="$mu" -v n="$CCF_NORM" '
        /^ccf_norm *:/       {print "ccf_norm : " n; next}
        /^remove_bis_spot:/  {print "remove_bis_spot: " r; next}
        /^convective_shift:/ {print "convective_shift: " cs; if (mu != "") print "dumusque_mu_scaling: " mu; next}
        {print}' "$3" > "$3.tmp" && mv "$3.tmp" "$3"
    if ! grep -q "^remove_bis_spot: ${remove}$" "$3" || ! grep -q "^convective_shift: ${cs}$" "$3" ||
       ! grep -q "^ccf_norm : ${CCF_NORM}$" "$3"; then
        echo "ERROR: no remove_bis_spot, convective_shift or ccf_norm line in $3"
        return 1
    fi
}

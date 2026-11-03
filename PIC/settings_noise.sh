#!/bin/bash
# Settings of sweep_noise.dag: the decade at 4096 px (60 rings, hmi.M_45s) for several noise levels sigma of the
# magnetograms, the active pixels |B| >= noise_th = sigma and |B| > B_th = 3 sigma (sunsim.sdo.classify_pixels), the
# spots at T_sp = 487 K, ccf_norm = 2 (raw ring CCFs) and the Dumusque bisector times mu on the spots (as
# sweep_45_dumusque_mu_rawccf.dag). Two kinds of faculae, KIND set before sourcing (the scripts take it as first
# argument), one parameter file each (lines: noise_th, B_th, I_th, A_th|B_sat, Temp; no comment: the submit files
# queue from them):
#   KIND=plage    : parameters_noise_plage.txt, A_th = 0 (every group of bright pixels is plage)
#   KIND=fraction : parameters_noise_fraction.txt, every bright pixel is min(|B|/mu/B_sat, 1) facula, only the
#                   B_sat > B_th of every sigma
# The sets of the identification are those of the lines of the file (identify_days.py --params: B_th goes with its
# noise_th, no combination of the two). Everything in ${BASE_N}; SATIRE_FF (satire_ff.py satire, made on the computer
# with the SATIRE masks, in the repository) for the comparison of the filling factors (satire_ff.py compare).
# Sourced by build_grids_noise.sh (no KIND), identify_noise.sh, aggregate_noise.sh, run_noise.sh and collect_noise.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

MAGNETOGRAM=45s                        # hmi.M_45s
SIZE=4096                              # pixels of the HMI images (full resolution)
N_RINGS=60                             # rings of the SunSim grid (write_conf uses it)
N_BLOCKS=80                            # identification: the days in N_BLOCKS jobs (= Queue of identify_noise.sub)
N_AGGREGATE=40                         # aggregation: the sets in N_AGGREGATE jobs (= Queue of aggregate_noise.sub)
CCF_NORM=2                             # raw ring CCFs
CONVECTIVE_SHIFT=1.0                   # amplitude of the Dumusque bisector
DUMUSQUE_MU_SCALING="0, 1"             # a, b: the bisector of a ring times a + b mu
BASE_N=${BASE}/noise                   # confs/, grids/, sets_<TAG>/, opt_rv_<TAG>/, results_<TAG>/
CONFS=${BASE_N}/confs
GRIDS=${BASE_N}/grids
SATIRE_FF=${PIPELINES}/satire_ff.csv  # the SATIRE filling factors of every day with a mask (satire_ff.py satire)

# the values of a column of a parameter file
values() { awk -F'[ ,\t]+' -v c="$2" 'NF >= 5 && $1 !~ /^#/ {print $c + 0}' "$1" | sort -g -u | tr '\n' ' '; }
TEMPS=$(values "${PIPELINES}/parameters_noise_plage.txt" 5; values "${PIPELINES}/parameters_noise_fraction.txt" 5)

if [ -n "$KIND" ]; then
    case "$KIND" in
        plage)    X_NAME=A_th;  X_OPT=--A-th;  FACULAE=plage ;;
        fraction) X_NAME=B_sat; X_OPT=--B-sat; FACULAE=fraction ;;
        *) echo "ERROR: KIND must be plage or fraction, got '$KIND'"; exit 1 ;;
    esac
    TAG=noise_${KIND}_m${MAGNETOGRAM}_size_${SIZE}_N${N_RINGS}
    SETS=${BASE_N}/sets_${TAG}         # <name>/ : simulation and stats.json of every line of the parameter file
    IDENT=${SETS}/ident                # <set label>/ : identification of every set of thresholds
    PARAMS=${PIPELINES}/parameters_noise_${KIND}.txt
    PARAM_COLUMNS=noise_th,B_th,I_th,${X_NAME},Temp     # the columns of PARAMS (identify_days.py --columns)
fi

# write_conf of settings.sh with the Dumusque bisector times mu and ccf_norm = CCF_NORM:  write_conf_noise <Temp> <file>
write_conf_noise() {
    write_conf "$1" "$2"
    awk -v cs="$CONVECTIVE_SHIFT" -v mu="$DUMUSQUE_MU_SCALING" -v n="$CCF_NORM" '
        /^ccf_norm *:/       {print "ccf_norm : " n; next}
        /^convective_shift:/ {print "convective_shift: " cs; print "dumusque_mu_scaling: " mu; next}
        {print}' "$2" > "$2.tmp" && mv "$2.tmp" "$2"
    if ! grep -q "^dumusque_mu_scaling" "$2" || ! grep -q "^ccf_norm : ${CCF_NORM}$" "$2"; then
        echo "ERROR: no convective_shift or ccf_norm line in $2"
        return 1
    fi
}

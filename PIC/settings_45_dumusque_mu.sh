#!/bin/bash
# Settings of the 45 s sweeps (4096 px, 60 rings, hmi.M_45s: settings_4096.sh for KIND=plage, settings_fraction.sh for
# KIND=fraction) with the bisector of Dumusque (2014) times mu added to the spots: [star] convective_shift =
# CONVECTIVE_SHIFT and dumusque_mu_scaling = DUMUSQUE_MU_SCALING (the bisector of a ring times a + b mu; 0, 1:
# proportional to mu, zero at the limb), instead of no bisector (convective_shift = 0 in settings.sh).
# The identification and the filling factors do not depend on the bisector: those of sweep_4096.dag and
# sweep_fraction.dag (IDENT of settings_4096.sh and settings_fraction.sh) are used, so these must have done their
# aggregate step first. Same parameter files (parameters.txt, parameters_fraction.txt); the confs, grids, sets, opt_rv
# and results in ${BASE_DMU}.
# CCF_NORM (environment, 0 if not set): ccf_norm of the configurations. 0 = -ccf/min(ccf) + 1, every ring normalised
# (sweep_45_dumusque_mu.dag); 2 = raw ring CCFs (mps_highres: the spots dark, the faculae bright and the quiet disc limb
# darkened in the RV too; sweep_45_dumusque_mu_rawccf.dag), with the confs, grids, sets, opt_rv and results in folders
# with the suffix _rawccf.
# Sourced by build_grids_45_dumusque_mu.sh (no KIND), run_opt_rv_45_dumusque_mu.sh and collect_stats_45_dumusque_mu.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC

case "$KIND" in
    "")       source "${PIPELINES}/settings_4096.sh" ;;      # SIZE, N_RINGS (the same in settings_fraction.sh)
    plage)    source "${PIPELINES}/settings_4096.sh";     X_NAME=A_th;  FACULAE=plage ;;
    fraction) source "${PIPELINES}/settings_fraction.sh"; X_NAME=B_sat; FACULAE=fraction ;;
    *) echo "ERROR: KIND must be plage or fraction, got '$KIND'"; exit 1 ;;
esac

CCF_NORM=${CCF_NORM:-0}
RUN=$([ "$CCF_NORM" = 2 ] && echo _rawccf)

# the Temp of the two parameter files (build_grids_45_dumusque_mu.sh); with CCF_NORM = 2 only plage is run: parameters.txt
PARAM_FILES="${PIPELINES}/parameters.txt"
[ "$CCF_NORM" = 2 ] || PARAM_FILES="$PARAM_FILES ${PIPELINES}/parameters_fraction.txt"
TEMPS=$(awk -F'[ ,\t]+' 'NF >= 4 && $1 !~ /^#/ {print $4 + 0}' $PARAM_FILES | sort -g -u | tr '\n' ' ')

CONVECTIVE_SHIFT=1.0                   # amplitude of the Dumusque bisector (old convective_shift)
DUMUSQUE_MU_SCALING="0, 1"             # a, b: the bisector of a ring times a + b mu
BASE_DMU=${BASE}/dumusque_mu           # confs/, grids/, sets_<TAG>/, opt_rv_<TAG>/, results_<TAG>/
CONFS=${BASE_DMU}/confs${RUN}
GRIDS=${BASE_DMU}/grids${RUN}

if [ -n "$KIND" ]; then
    SETS=${BASE_DMU}/sets_${TAG}${RUN} # <name>/ : simulation and stats.json; IDENT stays the one of the 45 s sweep
fi

# write_conf of settings.sh with the Dumusque bisector times mu and ccf_norm = CCF_NORM
write_conf_dumusque_mu() {
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

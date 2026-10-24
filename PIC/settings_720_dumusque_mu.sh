#!/bin/bash
# Settings of the 720 s sweep (settings_720.sh) with the bisector of Dumusque (2014) times mu added to the spots:
# [star] convective_shift = CONVECTIVE_SHIFT and dumusque_mu_scaling = DUMUSQUE_MU_SCALING (the bisector of a ring times
# a + b mu; 0, 1: proportional to mu, zero at the limb), instead of no bisector (convective_shift = 0 in settings.sh).
# The identification and the filling factors do not depend on the bisector: those of sweep_720.dag (IDENT of
# settings_720.sh) are used, so sweep_720.dag must have done its aggregate step first. Same parameter files
# (parameters_720_plage.txt, parameters_720_fraction.txt); the confs, grids, sets, opt_rv and results in ${BASE_DMU}.
# Sourced by build_grids_720_dumusque_mu.sh (no KIND), run_opt_rv_720_dumusque_mu.sh and collect_stats_720_dumusque_mu.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_720.sh"

CONVECTIVE_SHIFT=1.0                   # amplitude of the Dumusque bisector (old convective_shift)
DUMUSQUE_MU_SCALING="0, 1"             # a, b: the bisector of a ring times a + b mu
BASE_DMU=${BASE720}/dumusque_mu        # confs/, grids/, sets_<TAG>/, opt_rv_<TAG>/, results_<TAG>/
CONFS=${BASE_DMU}/confs
GRIDS=${BASE_DMU}/grids

if [ -n "$KIND" ]; then
    SETS=${BASE_DMU}/sets_${TAG}       # <name>/ : simulation and stats.json; IDENT stays the one of settings_720.sh
fi

# write_conf of settings.sh with the Dumusque bisector times mu
write_conf_dumusque_mu() {
    write_conf "$1" "$2"
    awk -v cs="$CONVECTIVE_SHIFT" -v mu="$DUMUSQUE_MU_SCALING" \
        '/^convective_shift:/ {print "convective_shift: " cs; print "dumusque_mu_scaling: " mu; next} {print}' \
        "$2" > "$2.tmp" && mv "$2.tmp" "$2"
    if ! grep -q "^dumusque_mu_scaling" "$2"; then
        echo "ERROR: no convective_shift line in $2"
        return 1
    fi
}

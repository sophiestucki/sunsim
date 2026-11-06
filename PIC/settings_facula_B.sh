#!/bin/bash
# Settings of sweep_facula_B.dag: the decade at 4096 px (60 rings, hmi.M_45s), the plage of one set of thresholds
# (noise_th, B_th, I_th, A_th) and T_sp, ccf_norm = 2 (raw ring CCFs) and the Dumusque bisector times mu on the spots
# (as sweep_network.dag), for several mean fields B_fc of the faculae: the MURaM facula intensities (the 200 G
# simulation) of every mu replaced by (1 - B_fc/200) I_quiet + B_fc/200 I_faculae ([star] facula_B of the
# configuration, sunsim.old_config; extrapolated linearly above 200 G, B_fc = 200: the faculae as before).
#   parameters_facula_B.txt : noise_th, B_th, I_th, A_th, B_fc, Temp (no comment: the submit files queue from it)
# The identification depends only on the thresholds (one set for all B_fc); one configuration and grids per
# (Temp, B_fc). Everything in ${BASE_FB}. Sourced by build_grids_facula_B.sh, identify_facula_B.sh,
# aggregate_facula_B.sh, run_facula_B.sh and collect_facula_B.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

MAGNETOGRAM=45s                        # hmi.M_45s
SIZE=4096                              # pixels of the HMI images (full resolution)
N_RINGS=60                             # rings of the SunSim grid (write_conf uses it)
N_BLOCKS=80                            # identification: the days in N_BLOCKS jobs (= Queue of identify_facula_B.sub)
CCF_NORM=2                             # raw ring CCFs
CONVECTIVE_SHIFT=1.0                   # amplitude of the Dumusque bisector
DUMUSQUE_MU_SCALING="0, 1"             # a, b: the bisector of a ring times a + b mu
FACULAE=plage
TAG=facula_B_m${MAGNETOGRAM}_size_${SIZE}_N${N_RINGS}
BASE_FB=${BASE}/facula_B               # confs/, grids/, sets_<TAG>/, opt_rv_<TAG>/, results_<TAG>/
CONFS=${BASE_FB}/confs
GRIDS=${BASE_FB}/grids
SETS=${BASE_FB}/sets_${TAG}            # <name>/ : simulation and stats.json of every line of the parameter file
IDENT=${SETS}/ident                    # <set label>/ : identification of every set of thresholds
PARAMS=${PIPELINES}/parameters_facula_B.txt
PARAM_COLUMNS=noise_th,B_th,I_th,A_th,B_fc,Temp     # the columns of PARAMS (identify_days.py --columns)

# names of the configuration and of the grids of a (Temp, B_fc):  conf_name_facula_B <Temp> <B_fc>
conf_name_facula_B() {
    echo "T_$(printf "%.0f" "$1")_Bfc_$(printf "%g" "$2")_N${N_RINGS}"
}

# write_conf of settings.sh with the Dumusque bisector times mu, ccf_norm = CCF_NORM and facula_B:
#   write_conf_facula_B <Temp> <B_fc> <file>
write_conf_facula_B() {
    write_conf "$1" "$3"
    awk -v cs="$CONVECTIVE_SHIFT" -v mu="$DUMUSQUE_MU_SCALING" -v n="$CCF_NORM" -v B="$2" '
        /^ccf_norm *:/       {print "ccf_norm : " n; next}
        /^convective_shift:/ {print "convective_shift: " cs; print "dumusque_mu_scaling: " mu; print "facula_B: " B; next}
        {print}' "$3" > "$3.tmp" && mv "$3.tmp" "$3"
    if ! grep -q "^dumusque_mu_scaling" "$3" || ! grep -q "^ccf_norm : ${CCF_NORM}$" "$3" ||
       ! grep -q "^facula_B: $2$" "$3"; then
        echo "ERROR: no convective_shift or ccf_norm line in $3"
        return 1
    fi
}

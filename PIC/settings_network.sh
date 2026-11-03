#!/bin/bash
# Settings of sweep_network.dag: the decade at 4096 px (60 rings, hmi.M_45s), T_sp = 500 K, ccf_norm = 2 (raw ring
# CCFs) and the Dumusque bisector times mu on the spots (as sweep_noise.dag), the time series of the parts simulated
# one by one to test whether the network (the bright groups smaller than A_th) behaves as diluted faculae
# (identify_network.py):
#   spot, plage (groups > A_th), network (groups of A_th or less with >= min_px pixels, full pixels), network_fraction
#   (the same pixels, min(|B| / mu / B_sat, 1) facula), spot_plage, spot_plage_network, spot_plage_network_fraction
# and the noise diagnostics of every (noise_th, B_th, I_th) of the parameter file and of DIAG_B_TH: the bright groups
# of every size in the data and in pure noise of the sigma(mu) and lag-1 correlation of every day.
#   parameters_network.txt : noise_th, B_th, I_th, A_th, min_px, B_sat, Temp (no comment: run_network.sub queues
#                            from it; min_px in pixels of the 4096 px images, 0: every small group is network)
# Everything in ${BASE_NW}. Sourced by build_grids_network.sh, identify_network.sh, aggregate_network.sh,
# run_network.sh and collect_network.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

MAGNETOGRAM=45s                        # hmi.M_45s
SIZE=4096                              # pixels of the HMI images (full resolution only)
N_RINGS=60                             # rings of the SunSim grid (write_conf uses it)
N_BLOCKS=80                            # identification: the days in N_BLOCKS jobs (= Queue of identify_network.sub)
CCF_NORM=2                             # raw ring CCFs
CONVECTIVE_SHIFT=1.0                   # amplitude of the Dumusque bisector
DUMUSQUE_MU_SCALING="0, 1"             # a, b: the bisector of a ring times a + b mu
DIAG_NOISE_TH=8                        # noise diagnostics also for B_th = DIAG_B_TH [G], with this noise_th, I_th
DIAG_B_TH="24 30 36 48"
DIAG_I_TH=0.9
DIAG_B_SAT=200                         # facular fraction of the bright groups of every size in the diagnostics [G]
TAG=network_m${MAGNETOGRAM}_size_${SIZE}_N${N_RINGS}
BASE_NW=${BASE}/network                # confs/, grids/, sets_<TAG>/, results_<TAG>/
CONFS=${BASE_NW}/confs
GRIDS=${BASE_NW}/grids
SETS=${BASE_NW}/sets_${TAG}            # <part>/<name>/ : simulation (and stats.json)
IDENT=${SETS}/ident                    # <set label>/ : cells of every day; diag/<key label>/ : noise diagnostics
PARTS_DIR=${SETS}/parts                # <set label>/<part>/ : filling factors of every part
PARAMS=${PIPELINES}/parameters_network.txt
PARTS="spot plage network network_fraction spot_plage spot_plage_network spot_plage_network_fraction"
TOTALS="spot_plage spot_plage_network spot_plage_network_fraction"   # compared with the observations (stats.json)

TEMPS=$(awk -F'[ ,\t]+' 'NF >= 7 && $1 !~ /^#/ {print $7 + 0}' "$PARAMS" | sort -g -u | tr '\n' ' ')

# label of a set (identify_network.py set_label):  set_label_network <noise_th> <B_th> <I_th> <A_th> <min_px> <B_sat>
set_label_network() {
    python -c "import sys; print('noise%g_B%g_I%g_A%g_min%g_Bsat%g' % tuple(map(float, sys.argv[1:])))" "$@"
}

# write_conf of settings.sh with the Dumusque bisector times mu and ccf_norm = CCF_NORM:  write_conf_network <Temp> <file>
write_conf_network() {
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

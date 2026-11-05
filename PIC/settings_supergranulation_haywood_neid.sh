#!/bin/bash
# Settings of the supergranulation jobs at the times of the NEID exposures (sourced by supergranulation_haywood_neid_day.sh
# and supergranulation_haywood_neid_collect.sh), on top of settings_supergranulation_haywood.sh (same HMI processing).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_supergranulation_haywood.sh"

SG_HARPSN_OUT=$SG_OUT                  # the HARPS-N days: with the NEID ones in the CLV drift fit of collect
NEID=${BASE}/observations/neid_solarl2_2021-05-01_2021-07-01.csv   # solarl2 table of the NEID archive
SG_OUT=${BASE}/supergranulation_haywood_neid_${SG_START}_${SG_END}  # days/<date>/ (one per line of supergranulation_haywood_neid_days.txt), results
SG_QUALITY="--min-snr-ratio 0.98 --min-day-exp 50 --harpsn-out ${SG_HARPSN_OUT} --harpsn ${HARPSN} --min-obs-quality 0.99"
# NEID clouds: extsnr >= 0.98 x the highest within +- 10 min; HARPS-N days: the cuts of its run
# collect: one results folder per variant (name: options of supergranulation_haywood.py collect)
SG_VARIANTS=(
  "results_2days_rotvr_clv5vr_h2_diffext_snr0.98:--rotation-obs-vr --clv-deg 5 --clv-drift obs_vr --drift-deg 2 --drift-harmonics 2 --diff-extinction"
)

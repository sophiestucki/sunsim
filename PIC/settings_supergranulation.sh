#!/bin/bash
# Settings of the supergranulation jobs (sourced by supergranulation_day.sh and supergranulation_collect.sh), on top of
# settings.sh (environment, HARPS-N and interventions files).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

SG_START=2021-05-01                    # the period (END excluded): one job per day of supergranulation_days.txt
SG_END=2021-08-01
SG_OUT=${BASE}/supergranulation_${SG_START}_${SG_END}   # days/<date>/ (products of every day), results/
SG_SIZE=4096                           # pixels of the HMI images (4096: full resolution)
SG_RINGS=30                            # rings of the SunSim grid (60 gave the same results on 2021-05-05 and 2021-05-31)
SG_THRESHOLDS="--noise-th 8 --B-th 24 --I-th 0.89 --A-th 20 --mu-area 0.1 --faculae plage"
SG_PMODES="--pmodes"                   # "" : no 45 s dopplergrams (~15 x fewer downloads; can be added later)
SG_WORKERS=2                           # downloads at the same time in one job
SG_CONF=${PIPELINES}/../conf/starsim_sun_sdo_mps_highres.conf
SG_MPS=/data/spotless/scratch/sstucki/starsim/mps_spectra/
SG_MASK=/nfs/pic.es/user/s/sstucki/env/starsim/starsim/masks/HARPS_G20_mps_friendly.mas

#!/bin/bash
# Settings of the supergranulation jobs with the dopplergrams corrected as in Haywood et al. (2016) (sourced by
# supergranulation_haywood_day.sh and supergranulation_haywood_collect.sh), on top of settings.sh (environment,
# HARPS-N and interventions files).
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

SG_START=2021-05-01                    # the period (END excluded): one job per day of supergranulation_haywood_days.txt
SG_END=2021-08-01
SG_OUT=${BASE}/supergranulation_haywood_${SG_START}_${SG_END}   # days/<date>/ (products of every day), results/
SG_SIZE=4096                           # pixels of the HMI images (4096: full resolution)
SG_RINGS=30                            # rings of the SunSim grid
SG_THRESHOLDS="--noise-th 8 --B-th 24 --I-th 0.89 --A-th 20 --mu-area 0.1 --faculae plage"
SG_WINDOW=720                          # [s] 45 s dopplergrams averaged around every exposure (p modes)
SG_MU_MIN=0.1                          # only the pixels with mu > this are projected (Haywood et al. 2016)
SG_WORKERS=2                           # downloads at the same time in one job
SG_CONF=${PIPELINES}/../conf/starsim_sun_sdo_mps_highres.conf
SG_MPS=/data/spotless/scratch/sstucki/starsim/mps_spectra/
SG_MASK=/nfs/pic.es/user/s/sstucki/env/starsim/starsim/masks/HARPS_G20_mps_friendly.mas

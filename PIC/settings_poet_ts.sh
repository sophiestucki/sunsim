#!/bin/bash
# Settings of the identification at the times of the PoET disk-integrated time series (poet_ts.dag), on top of
# settings.sh (environment, BASE). Sourced by poet_ts_day.sh and poet_ts_collect.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

POET_TIMES=${PIPELINES}/poet_ts_jdb.csv        # column jdb: time_jdb of PoET_disk-integrated_time_series (BJD TDB)
POET_START=2026-04-07                          # t = 0 of the filling factors (00:00 UTC): the first day of POET_TIMES
POET_OUT=${BASE}/poet_ts                       # days/, ident/, exposures.csv (poet_epochs.py)
PARAMS=${PIPELINES}/parameters_poet_ts.txt     # noise_th, I_th, A_th, Temp of CAUP/poet_sdo_s1d_a.ipynb
SIZE=4096                                      # pixels of the HMI images (full resolution)
N_RINGS=60                                     # rings of the cell filling factors
MAX_DT=720                                     # [s] exposures farther than this from a 720 s continuum are left out

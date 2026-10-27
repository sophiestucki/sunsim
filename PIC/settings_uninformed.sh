#!/bin/bash
# Settings of uninformed_decade.dag: the statistical model of the masks of uninformed_masks_model.ipynb (starsim
# codes+tests) trained on the decade (START to END of settings.sh), with the SDO masks of noise_th = 8, I_th = 0.8,
# A_th = 10 at 1024 px from the hmi.M_45s magnetograms (THRESHOLDS and SIZE of uninformed_decade.py), then 3 months of
# synthetic masks at the solar minimum and maximum. Everything in ${BASE_U}:
#   maps/       the SDO masks (spot_map_<t>.npz, faculae_map_<t>.npz, days/<date>.json, summary.csv)
#   model/      model.json (the parameters of the generator), statistics.pkl, bins.csv, features.pkl, train.log
#   synthetic/  <min|max>_<k>/ (the synthetic masks), statistics.pkl, synth.log
# Sourced by uninformed_maps.sh, uninformed_train.sh and uninformed_synth.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings.sh"

BASE_U=${BASE}/uninformed_decade
N_BLOCKS=40                            # maps: the days in N_BLOCKS jobs (= Queue of uninformed_maps.sub)
SYN_MIN=2019-11-01                     # first day of the 3 months at the minimum
SYN_MAX=auto                           # at the maximum: auto = the 91 days of highest spot filling factor

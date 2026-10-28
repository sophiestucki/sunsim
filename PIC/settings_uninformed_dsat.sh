#!/bin/bash
# Settings of uninformed_decade_dsat.dag: the training and the synthetic 3 months of uninformed_decade.dag again, with
# the decay rate of the regions constant above A_SAT MSH (uninformed_decade.py train --decay-sat; the decay law is
# fitted below A_SAT only), on the SDO masks already in ${BASE_U}/maps (not computed again). The results of
# uninformed_decade.dag are left as they are; everything new in ${BASE_D}:
#   model/      model.json, statistics.pkl, bins.csv, train.log, features.pkl (copied from ${BASE_U}/model)
#   synthetic/  <min|max>_<k>/ (the synthetic masks), statistics.pkl, synth.log
# Sourced by uninformed_train_dsat.sh and uninformed_synth_dsat.sh.
PIPELINES=/nfs/pic.es/user/s/sstucki/sunsim/PIC
source "${PIPELINES}/settings_uninformed.sh"

BASE_D=${BASE}/uninformed_decade_dsat
MAPS_U=${BASE_U}/maps                  # the SDO masks of uninformed_decade.dag
A_SAT=100                              # MSH, decay rate D0 (min(A, A_SAT) / 100)^decay_p

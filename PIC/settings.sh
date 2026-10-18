#!/bin/bash
# Settings shared by build_grids.sh and run_10_opt_rv.sh (sourced by both): folders, period, environment, and
# write_conf <Temp> <file>, the configuration of the simulations (only spot_T_contrast changes: one grid per Temp).

BASE=/data/spotless/scratch/sstucki/sunsim              # outputs: confs/, grids/, sets/, opt_rv/
SDO_DATA=/data/spotless/scratch/sstucki/starsim/SDO_data/  # HMI fits
START=2015-07-18
END=2024-01-29
SIZE=512

# observations for the statistics of every set (python -m sunsim.observations stats)
HARPSN=/data/spotless/scratch/sstucki/starsim/observations/harpn_sun_release_timeseries_2015-2025.csv
INTERVENTIONS=/data/spotless/scratch/sstucki/starsim/observations/harpsn_intervention_days.txt
SORCE="/data/spotless/scratch/sstucki/starsim/observations/DataSSI - SORCE.sav"
TSIS=/data/spotless/scratch/sstucki/starsim/observations/tsis_tsi_24hr.csv
WINDOW=""          # e.g. WINDOW="--window 2021-01-01 2024-01-01": only these epochs in the statistics

source /nfs/pic.es/user/s/sstucki/env/simu/bin/activate    # needs sunsim installed: pip install "sunsim[sdo]"

write_conf() {
    local Temp=$1 file=$2
    mkdir -p "$(dirname "$file")"
    cat > "$file" << EOF
[files]

filter_name : TJO_R.dat
orders_CRX_filename: orders_wv_CARMENES.dat

[general]

simulation_mode: grid
n_grid_rings : 15
wavelength_lower_limit : 3801.0
wavelength_upper_limit : 7999.0
spectral_library : mps_highres
mps_spectra_folder : /data/spotless/scratch/sstucki/starsim/mps_spectra/
stagger_spectra_folder : /home/sophie-stucki/Documents/starsim_simulations/nina_spec/stagger_spectra/
mps_mu_grid : 0
bb_ratio : 1
ccf_norm : 0
flux_flat : 0
highres_mu : 1
smoothing_kernel: 50001
remove_bis_spot: 1
add_median_diff: 0

[rv]

ccf_template: mask
ccf_mask: /nfs/pic.es/user/s/sstucki/env/starsim/starsim/masks/HARPS_G20_mps_friendly.mas
ccf_weight_lines: 0
path_weight_lines: flux_ratios.dat
ccf_rv_range : 18.0
ccf_rv_step : 0.25
instrument: None

[star]

temperature_photosphere : 5778.0
spot_T_contrast : ${Temp}
facula_T_contrast : 30.0
facular_area_ratio : 0.0
convective_shift: 0.0
rotation_period : 25.05
radius : 1
mass : 1
logg : 4.4
inclination : 90.0
differential_rotation : 0

[LD]

use_phoenix_limb_darkening : 1
limb_darkening_law : linear
limb_darkening_q1: 0.0
limb_darkening_q2: 0.0
limb_extrapolation: constant
limb_lim: 0.1
meunier : 1

[spots]

plot_grid_map : 0
spots_evo_law : constant
reference_time : 0.0
sdo_input : 1
SDO_input_path : ${BASE}/sets/

[planet]

simulate_planet: 0
planet_period : 0.932
planet_transit_t0 : 2.0
planet_radius : 0.07
planet_impact_param : 0.0
planet_esinw: 0.0
planet_ecosw: 0.0
planet_spin_orbit_angle : 0.0
planet_semi_amplitude: 1.0
EOF
}

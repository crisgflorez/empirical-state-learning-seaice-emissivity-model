#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=nf
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=seaice_finalV2
#SBATCH -o sbatchlogs/seaiceV2_bg_emis07_with_losses_initial_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_25epochs_sbatch_python3_8%j.out
#SBATCH -e sbatchlogs/seaiceV2_bg_emis07_with_losses_initial_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_25epochs_sbatch_python3_8%j.err

#module load python3/3.12.9-01
module load python3/3.8.8-01

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/model
pyScript=seaice_training.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript} \
  --data /perm/dnk8355/paper2026/netcdf_1april2024_31march2026 \
  --sensors METOP-C \
  --output /perm/dnk8355/paper2026/outputs_training/exp1_METOPC_obs_err_fromMETOPB \
  --tag 1april2024_31march2026_bg_emis07_with_losses_initial_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_25epochs_sbatch_python3_8 \
  --reproducible


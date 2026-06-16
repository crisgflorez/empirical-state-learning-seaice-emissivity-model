#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=nf
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=seaice_finalV2
#SBATCH -o sbatchlogs/seaiceV2_bg_emis08_with_losses_original_obs_errors_bg_biasice2_5_ocean5_bg_bias_err0_001_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10%j.out
#SBATCH -e sbatchlogs/seaiceV2_bg_emis08_with_losses_original_obs_errors_bg_biasice2_5_ocean5_bg_bias_err0_001_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10%j.err

#module load python3/3.8.8-01 
module load python3/3.10.10-01


sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model
pyScript=seaice_training.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript} \
  --data /perm/dnk8355/netcdf_1april2024_31march2025 \
  --sensors METOP-B \
  --output /perm/dnk8355/outputs_training_v2_jan26_report_final \
  --tag 1april2024_31march2025_bg_emis08_with_losses_original_obs_errors_bg_biasice2_5_ocean5_bg_bias_err0_001_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10 \
  --reproducible


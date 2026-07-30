#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=ng #gf
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus=1
#SBATCH --job-name=seaice_gpu
#SBATCH -o sbatchlogs/seaice_gpu_expA_METOPC_8epochs.out
#SBATCH -e sbatchlogs/seaice_gpu_expA_METOPC_8epochs.err

module load python3/may23

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/model
pyScript=seaice_training.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript} \
  --data /perm/dnk8355/paper2026/netcdf_1april2024_31march2026 \
  --sensors METOP-C \
  --output /perm/dnk8355/paper2026/outputs_training/expA_METOPC \
  --tag 1april2024_31march2026_bg_emis07_with_losses_adjusted_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_002_newimplementation_in_emisNN_with_angle_8epochs_sbatch_python3_10 \
  --nepochs 8 \
  --initial_seaice ifs_seaice_initials_METOP-C_1apr2024_31march2026_without_land_without_nans.nc \
  --initial_tsfc ifs_tsfc_METOP-C_1apr2024_31march2026_dailyx_without_land.nc \
  --reproducible \
  # --trainonly 
  # --zenith_as_predictor
  # --pretrained_model #activate only if i want to use a pretrained model


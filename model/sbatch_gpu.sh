#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=ng #gf
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus=1
#SBATCH --job-name=seaice_gpu
#SBATCH -o sbatchlogs/seaice_gpu_metopB_3epochs.out
#SBATCH -e sbatchlogs/seaice_gpu_metopB_3epochs.err

module load python3/may23

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/model
pyScript=seaice_training.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript} \
  --data /perm/dnk8355/paper2026/netcdf_1april2024_31march2026 \
  --sensors METOP-B \
  --output /perm/dnk8355/paper2026/outputs_training/exp3_METOPB \
  --tag 1april2024_31march2026_bg_emis07_with_losses_adjusted_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_002_newimplementation_in_emisNN_with_angle_3epochs_sbatch_python3_10_pretrained_model \
  --reproducible \
  --pretrained_model True


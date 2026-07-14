#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=ng #gf
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus=1
#SBATCH --job-name=seaice_gpu
#SBATCH -o sbatchlogs/seaice_gpu.out
#SBATCH -e sbatchlogs/seaice_gpu.err

module load python3/may23

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/model
pyScript=seaice_training.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript} \
  --data /perm/dnk8355/paper2026/netcdf_1april2024_31march2026 \
  --sensors METOP-C \
  --output /perm/dnk8355/paper2026/outputs_training/exp1_METOPC_obs_err_fromMETOPB \
  --tag seaice_gpu \
  --reproducible


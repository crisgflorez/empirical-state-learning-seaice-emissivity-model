#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=nf
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=seaice_finalV2
#SBATCH -o sbatchlogs/seaiceV2_%j.out
#SBATCH -e sbatchlogs/seaiceV2_%j.err

module load python3/3.10.10-01


sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model
pyScript=test.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript} \
  --data /perm/dnk8355/netcdf_1april2024_31march2025 \
  --sensors METOP-B \
  --output /perm/dnk8355/outputs_training_finalv2 \
  --tag _1april2024_31march2025 \
  --diagsonly
  --trainonly
  --reproducible
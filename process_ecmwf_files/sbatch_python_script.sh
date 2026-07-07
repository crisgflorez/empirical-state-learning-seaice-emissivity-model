#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # pide 4 CPUs para esta tarea
#SBATCH --mem=16G               # pide 16 GB de memoria
#SBATCH --job-name=daily_nc_per_satellite
#SBATCH --output=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/sbatchlogs/daily_odb_to_daily_nc_per_satellite%j.out
#SBATCH -e /home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/sbatchlogs/daily_odb_to_daily_nc_per_satellite%j.err

module load python3/3.10.10-01

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model
pyScript=daily_odb_to_daily_nc_per_satellite.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript}



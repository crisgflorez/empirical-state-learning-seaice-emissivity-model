#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # ask 4 CPUs para esta tarea
#SBATCH --mem=16G               # ask 16 GB de memoria
#SBATCH --job-name=daily_nc_per_satellite
#SBATCH --output=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/sbatchlogs/merging_daily_netcdfiles_into_monthly_netcdf%j.out
#SBATCH -e /home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/sbatchlogs/merging_daily_netcdfiles_into_monthly_netcdf%j.err

module load python3/3.10.10-01

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/process_ecmwf_files
pyScript=merging_daily_netcdfiles_into_single_netcdf.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript}



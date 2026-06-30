#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=nf
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # pide 4 CPUs para esta tarea
#SBATCH --mem=32G               # pide 16 GB de memoria
#SBATCH --job-name=check_nc_file
#SBATCH --output=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/sbatchlogs/check_nc_file_above_below45degrees%j.out
#SBATCH -e /home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/sbatchlogs/check_nc_file_above_below45degrees%j.err

module load python3/3.10.10-01

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model
pyScript=check_nc_files.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript}


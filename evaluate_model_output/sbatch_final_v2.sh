#!/bin/bash

###################################
# change here to submit the job
###################################
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=seaice_finalV2
#SBATCH -o sbatchlogs/statistics_METOPC_obs_err_fromMETOPB.out
#SBATCH -e sbatchlogs/statistics_METOPC_obs_err_fromMETOPB.err

#module load python3/3.12.9-01
module load python3/3.8.8-01

sourcePath=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/evaluate_model_output
pyScript=statistics_for_each_exp.py

echo "starting ${pyScript} ...."
python3 ${sourcePath}/${pyScript}
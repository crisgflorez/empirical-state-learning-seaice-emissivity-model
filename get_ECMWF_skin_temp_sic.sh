#!/bin/bash

#SBATCH --qos=np
#SBATCH --time=12:00:00 #1month of files about 2h 30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/sbatchlogs/grib_NH_SH.out
#SBATCH --job-name=grib_NH_SH


archDir=grib_files_NH_SH

if [ ! -d "$PERM/paper2026/${archDir}" ]; then
  mkdir -p "$PERM/paper2026/${archDir}"
fi

###################
cd $PERM/paper2026/${archDir}  
###################

echo "#####################"
echo "my dir is: $PERM/paper2026/${archDir}"
echo $(date) "- Start MARS retrieval" 
echo "#####################"
echo " " 

# configuration
only_one_day=false  # Change to true to download only one day
single_day="2024-03-14"  # Specific day if only_one_day=true

start_date="2024-04-01"
switch_date="2024-11-13" #When the model was changed from pre-operational to operational. 12/11/2024 last day that was pre-operational
end_date="2026-04-01"

if [ "$only_one_day" = true ]; then
  current_date="$single_day"
  end_date="$single_day"
else
  current_date="$start_date"
fi



while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
  YYYY=$(date -d "$current_date" +%Y)
  MM=$(date -d "$current_date" +%m)
  DD=$(date -d "$current_date" +%d)

  # Define expID according to the date
  # 1:  operational
  # 79: pre-operational 
  if [[ "$current_date" < "$switch_date" ]]; then
    expID=79
  else
    expID=1
  fi 


  echo "$current_date | expID: $expID"

#############################
# Configure ODB MARS REQUEST
#############################
# change targetFile name here
targetFile=HRES_SIC_TSKIN_N80_${YYYY}${MM}${DD}_NH_SH.grb  #_filtered.odb


cat > marsODB.inp << EOF
RETRIEVE,
    STREAM=oper,
    CLASS=od,
    EXPVER=${expID},
    GRID=N80,
    GAUSSIAN=reduced,
    TYPE=AN,
    DATE=${YYYY}${MM}${DD},
    TIME=00:00:00/06:00:00/12:00:00/18:00:00,
    LEVTYPE=SFC,
    PARAM=172.128/31.128/235.128,   #172.128: land sea mask, #31.128:sea ice area fraction, #235.128: Skin temperature
    FILTER="select * where (lat<-45 or lat>45)",
    TARGET=${targetFile}

EOF

# submit mars request
mars marsODB.inp


rm marsODB.inp
current_date=$(date -I -d "$current_date + 1 day")
done
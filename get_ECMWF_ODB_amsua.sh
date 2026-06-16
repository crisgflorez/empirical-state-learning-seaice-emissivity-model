#!/bin/bash

#SBATCH --qos=np
#SBATCH --time=12:00:00 #1month of files about 2h 30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=retrieve_odb_files_paper2026.out
#SBATCH --job-name=retrieve_odb_files_paper2026


archDir=odb_files

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
only_one_day=false    # Change to true to download only one day
single_day="2025-02-13"  # Specific day if only_one_day=true

start_date="2024-04-01"
switch_date="2024-11-13"
end_date="2026-04-01"       #$(date +%Y-%m-%d)

REPORTYPE=21009/21010
#21009	METOP-B AMSUA 	Radiances All-sky	AMSUA All-sky	Operational	2013-11-19	40r1
#21010	METOP-C AMSUA 	Radiances All-Sky	AMSUA All-sky	Operational	2018-11-08	46r1


# stream=oper --> for the early delivery (stream=oper) then "LSCREEN" is true in every outer loop
stream=lwda

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
targetFile=amsua_ofb_${stream}_${YYYY}${MM}${DD}_NH_SH_variables_filtered_lsm_lower02.odb  


cat > marsODB.inp << EOF
RETRIEVE,
    STREAM=${stream},
    CLASS=od,
    EXPVER=${expID},
    TYPE=OFB,
    DATE=${YYYY}${MM}${DD},
    TIME=00/12,
    REPORTYPE=${REPORTYPE},
    FILTER="select distinct expver, andate, antime, reportype,date,lsm,seaice, time, lat, lon,gp_number, satellite_identifier,satellite_instrument,windspeed10m,tsfc,snow_depth, snow_density,fg_rttov_cld_fraction,zenith,azimuth,scanline,scanpos,vertco_reference_1, emis_atlas, emis_retr, emis_fg, datum_tbflag,obsvalue,datum_status,datum_event1,datum_anflag,datum_rdbflag, biascorr, biascorr_fg, fg_depar, an_depar, final_obs_error,obs_error, tausfc, tup, tdown, tausfc_cld, tup_cld, tdown_cld  where lsm < 0.2 and (lat > 50 or lat<-50)",
    TARGET=${targetFile}

EOF

# submit mars request
mars marsODB.inp


rm marsODB.inp
current_date=$(date -I -d "$current_date + 1 day")
done
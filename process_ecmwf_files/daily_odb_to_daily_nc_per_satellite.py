import os
import pyodc as odc
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from useful_functions import datetime_to_julian
import xarray as xr
from sklearn.neighbors import BallTree
from multiprocessing import Pool

# -----------------------------
# 1. Load ODB paths and reference grid N80
# -----------------------------
base_path = '/perm/dnk8355/paper2026/odb_files/'
pattern = os.path.join(base_path, 'amsua_ofb_lwda_*_NH_SH_variables_filtered_lsm_lower02.odb')

all_files = sorted(glob.glob(pattern))

start = datetime(2024, 4, 1)
end   = datetime(2026, 3, 31)

filtered_files = []
for f in all_files:
    # Extract date from filename
    date_str = os.path.basename(f).split("_")[3]  # '20240313'
    date = datetime.strptime(date_str, "%Y%m%d")
    
    if start <= date <= end:
        filtered_files.append(f)

# -----------------------------
# CONFIG: which variables to export
# -----------------------------
# Set to "all" to process everything
# Or set to a list like ["FG_DEP"] or ["TSFC", "FG_DEP"]
export_vars = "all"
#export_vars = ["FG_DEP"]
# export_vars = ["TSFC", "FG_DEP"]
# CONFIG: skip writing text files (summaries + checks log)
skip_txt = False  # True = skip , False = write

# Load grid
grid_ds = xr.open_dataset("/perm/dnk8355/paper2026/grib_files_NH_SH/lat_lon_corrected_ref_above&below44degrees.nc")
lons_ref = grid_ds.lon.values
lats_ref = grid_ds.lat.values
grid_coords_rad = np.radians(np.column_stack((lats_ref, lons_ref)))

output_dir = '/perm/dnk8355/paper2026/netcdf_daily_1april2024_31march2026'
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 2. Processing function
# -----------------------------
def process_odb_file_daily(odbFile_name,output_dir, grid_coords_rad,export_vars,skip_txt):
    cols_needed = [
        'reportype','seaice@modsurf','lsm@modsurf','lat@hdr','lon@hdr',
        'time@hdr','date@hdr','vertco_reference_1@body',
        'obsvalue@body','tsfc@modsurf','windspeed10m@modsurf','emis_atlas@radiance_body',
        'fg_rttov_cld_fraction@allsky','zenith@sat', 'azimuth@sat',
        'scanline@radiance', 'scanpos@radiance','fg_depar@body','an_depar@body',
        'tausfc@radiance_body','tausfc_cld@allsky_body',
        'tup@radiance_body','tup_cld@allsky_body',
        'tdown@radiance_body','tdown_cld@allsky_body'
    ] #'an_depar@body',

    myODB = odc.read_odb(odbFile_name, single=True, columns=cols_needed)

    # Map reportype to satellite index
    myODB['reportype'] = myODB['reportype'].replace({
        21009: 0, #'METOP-B',
        21010: 1, #'METOP-C'
    })

    # Apply land-sea mask filter
    myODB = myODB[myODB['lsm@modsurf'] == 0] #< 0.01

    #invalid values in the dataframe are saved as -3.4028234663852886e+38
    # I could have used any other variable to identify the value of invalid values, 
    # but I chose emis_atlas@radiance
    invalid_value = myODB['emis_atlas@radiance_body'].min()

    # Filter channels of interest
    myODB = myODB[myODB['vertco_reference_1@body'].isin([1, 2, 3, 4, 15])]
    # We reindex the DataFrame to ensure it is continuous
    myODB = myODB.reset_index(drop=True)

    # Special variables that should not have invalid values
    special_vars = ['reportype', 'lsm@modsurf','lat@hdr','lon@hdr',
        'time@hdr','date@hdr','vertco_reference_1@body']
    other_vars = [col for col in cols_needed if col not in special_vars]

    # Create copy
    myODB_cleaned = myODB.copy()

    #For each row we put nans in all the "other_vars" columns
    #if at least one of these ones has an invalid value
    invalid_rows = (myODB_cleaned[other_vars] == invalid_value).any(axis=1)
    myODB_cleaned.loc[invalid_rows, other_vars] = np.nan  

    # In previous analysis I identified cases in which emis_atlas@radiance 
    # has an invalid value but not the rest of the variables, but these cases were very few.
    # So I decided to treat these cases the same way as the rest of the invalid values (removing the whole row).

   # Count duplicated rows in the dataframe (just for information)
   # For the moment we do not delete them, we keep them because 
   # we want to keep all the channels for each observation.
   # and sometimes only the row of one channel is duplicated later in the dataframe.
    total_rows = len(myODB_cleaned)
    mask_duplicates = myODB_cleaned.duplicated(keep=False)
    duplicated_rows = (mask_duplicates.sum())/2

    # We check if vertco_reference_1@body has the values 1, 2, 3, 4, 15 in each block of 5 rows
    # This is to ensure that we have all channels for each block.
    # These blocks define one observation, so we expect to have 5 rows per observation.
    pattern = [1, 2, 3, 4, 15]  # The expected pattern for vertco_reference_1@body
    pos = 0
    errors = []

    for idx, value in enumerate(myODB_cleaned['vertco_reference_1@body']):
        if value == pattern[pos]:
            pos = (pos + 1) % len(pattern)  # we move to the next position in the pattern
        else:
            errors.append(idx)  # we save the index of the error
            pos = 0  # reset the pattern if it doesn't match

    # We remove the rows not following the pattern
    # These rows are considered isolated rows that do not belong to any observation block
    if errors:
        myODB_cleaned = myODB_cleaned.drop(index=errors).reset_index(drop=True)
        print(f"[INFO] Removed {len(errors)} rows with invalid vertco_reference_1@body values.")

    myODB_cleaned['date@hdr'] = myODB_cleaned['date@hdr'].astype(str)
    myODB_cleaned['time@hdr'] = myODB_cleaned['time@hdr'].astype(str).str.zfill(6)
    myODB_cleaned['datetime'] = pd.to_datetime(myODB_cleaned['date@hdr'] + myODB_cleaned['time@hdr'], format='%Y%m%d%H%M%S')
    # Convert datetime to Julian day
    myODB_cleaned['JULIAN_DAY'] = datetime_to_julian(myODB_cleaned['datetime'])

    # We apply and extra check to ensure that the not channel dependent variables have the same value for each block of 5 rows
    # Define constant columns to check
    constant_cols = ['datetime','reportype','lsm@modsurf','seaice@modsurf',
                    'lat@hdr','lon@hdr','tsfc@modsurf','windspeed10m@modsurf',
                    'fg_rttov_cld_fraction@allsky','zenith@sat','azimuth@sat', 
                    'scanline@radiance', 'scanpos@radiance']

    # Create a block column to group every 5 rows together
    myODB_cleaned['block'] = myODB_cleaned.index // 5

    # Check if all constant columns have the same value within each block
    check_constants = (
        myODB_cleaned.groupby('block')[constant_cols]
        .nunique() #.nunique() ignores NaN values
        .max(axis=1)
    )
    # We identify the blocks with issues
    # A block is considered bad if the variables that are not channel dependent
    # do not have the same value for all the rows in the block
    bad_blocks_constants = check_constants[check_constants > 1].index

    if not skip_txt:
        # Append in to log file
        log_dir = "/perm/dnk8355/paper2026/odb_files/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, "checks_odb_files.txt")

        with open(log_filename, "a") as f:
            f.write(f"File: {os.path.basename(odbFile_name)} | "
                    f"total_nb_rows_with_filtering_but_without_removing_isolated_rows: {total_rows} | "
                    f"Nb.rows_with_at_least_1_invalid_column: {invalid_rows.sum()} | "
                    f"Percent.rows_with_at_least_1_invalid_column: {(invalid_rows.sum()/ total_rows)*100:.4f} | "
                    f"Duplicated rows: {duplicated_rows}  |"
                    f"Percent.dupl: {(duplicated_rows / total_rows)*100:.4f} |"
                    f"Nb.isolated_rows: {len(errors)}  |"
                    f"Total bad blocks: {len(bad_blocks_constants)} | "
                    f"Percent.bad blocks: {(len(bad_blocks_constants)/max(myODB_cleaned['block'].unique()))*100:.4f} \n")

    #total_nb_rows_with_filtering here filtering refers to the the selection of the channels of interest
    # and the land-sea mask filtering, but not the removal of isolated rows in case there are any.

    # Convert lat/lon to radians for BallTree
    sat_coords_rad = np.radians(np.column_stack((myODB_cleaned['lat@hdr'].values, myODB_cleaned['lon@hdr'].values)))
    # Create a BallTree for fast nearest neighbor search
    # Using haversine metric which requires coordinates in radians
    # grid_coords_rad is the reference grid coordinates reduced gaussian N80 in radians
    tree = BallTree(grid_coords_rad, metric='haversine')
    _, indices = tree.query(sat_coords_rad, k=1)

    myODB_cleaned['igrid_ini'] = indices.flatten()
    myODB_cleaned['nearest_lats'] = lats_ref[indices.flatten()]
    myODB_cleaned['nearest_lons'] = lons_ref[indices.flatten()]

    # Define variable mappings
    # The mapping is from odb variable names to NetCDF variable names
    var_mapping = {
        'lat@hdr': 'LAT','lon@hdr': 'LON','JULIAN_DAY': 'JULIAN_DAY','igrid_ini': 'INITIAL_IGRID',
        'obsvalue@body': 'OBSVALUE','tsfc@modsurf': 'TSFC','windspeed10m@modsurf': 'WINDSPEED10M',
        'fg_rttov_cld_fraction@allsky': 'CLOUD_FRACTION','emis_atlas@radiance_body': 'EMIS_WATER',
        'tausfc@radiance_body': 'TAUSFC','tdown@radiance_body': 'TDOWN','tup@radiance_body': 'TUP',
        'tausfc_cld@allsky_body': 'TAUSFC_CLD','tdown_cld@allsky_body': 'TDOWN_CLD','tup_cld@allsky_body': 'TUP_CLD',
        'nearest_lats': 'NEAREST_LATS','nearest_lons': 'NEAREST_LONS','seaice@modsurf': 'SEAICE',
        'zenith@sat': 'ZENITH','azimuth@sat': 'AZIMUTH',
        'scanline@radiance': 'SCANLINE', 'scanpos@radiance': 'SCANPOS', 'fg_depar@body': 'FG_DEP','an_depar@body': 'AN_DEP'
    }

    # Variables independent of the channel
    one_d_vars = ['JULIAN_DAY','INITIAL_IGRID','TSFC','WINDSPEED10M','CLOUD_FRACTION','LAT','LON','NEAREST_LATS','NEAREST_LONS','SEAICE','ZENITH','AZIMUTH','SCANLINE','SCANPOS']
    # Variables that depend on the channel
    # We will save these variables in a 2D array with shape (obs, channel)
    chan_vars = ['OBSVALUE','EMIS_WATER','TAUSFC','TDOWN','TUP','TAUSFC_CLD','TDOWN_CLD','TUP_CLD','FG_DEP','AN_DEP']
    
    # -----------------------------
    # Export loop
    # -----------------------------
    if export_vars == "all":
        target_one_d_vars = one_d_vars
        target_chan_vars = chan_vars
    else:
        target_one_d_vars = [v for v in one_d_vars if v in export_vars]
        target_chan_vars = [v for v in chan_vars if v in export_vars]

    date_str = max(myODB_cleaned['datetime']).strftime("%Y%m%d")

    def filter_valid_blocks(sat_data, n_channels):
        """
        Delete blocks with NaNs in any of the channels.
        """
        has_invalid_array = sat_data['has_invalid'].values
        has_invalid_reshaped = has_invalid_array.reshape(-1, n_channels)
        valid_blocks_mask = ~np.any(has_invalid_reshaped, axis=1)
        return sat_data[valid_blocks_mask.repeat(n_channels)].reset_index(drop=True)

    # We will create a NetCDF file for each satellite and each variable
    # The output files will be named as: <satellite>_<date>_<variable>.nc
    # e.g. NOAA-15_20240501_OBSVALUE.nc
    names_sat=['METOP-B', 'METOP-C']
    for sat in myODB_cleaned['reportype'].unique():
        sat_data = myODB_cleaned[myODB_cleaned['reportype'] == sat].copy()
        sat_name = names_sat[sat]

        # We identify rows in each satelite data with invalid values in the other variables
        # Basically it is a way to check which channels have NaN 
        # as previously we had already set NaN for all the variables in that channel if one of them had an invalid value
        # Add a flag to the DataFrame
        sat_data['has_invalid'] = sat_data[other_vars].isna().any(axis=1)

        # Define grouping keys to represent one observation (ignoring channels)
        #group_keys = ['reportype', 'date@hdr','time@hdr', 'lat@hdr', 'lon@hdr']
        #dup_check = sat_data.groupby(group_keys + ['vertco_reference_1@body']).size()
        #duplicates = dup_check[dup_check > 1]
        #print(duplicates)

        # Flag each observation group as invalid if any channel row is invalid.
        # Here invalid is associated with having NaN in any of the channels not
        # with having NaN in any of the variables in the other_vars list in each row
        #invalid_obs_flags = sat_data.groupby(group_keys)['has_invalid'].any()
        #n_invalid_obs = invalid_obs_flags.sum()
        has_invalid_array = sat_data['has_invalid'].values
        has_invalid_reshaped = has_invalid_array.reshape(-1, 5)
        obs_with_invalid = np.any(has_invalid_reshaped, axis=1) #it returns True if at least one channel in the block of 5 rows has NaN otherwise False
        n_invalid_obs = np.sum(obs_with_invalid)

        # Filter rows in sat_data that have invalid values in one of the channels
        nan_rows=sat_data[sat_data['has_invalid']] #select rows where has_invalid is True
        # Count NaN values per channel
        counts_by_channel = nan_rows['vertco_reference_1@body'].value_counts().sort_index()

        if not skip_txt:
            # Prepare output path
            sat_output_txt = os.path.join(output_dir, f"{sat_name}_{date_str}_summary.txt")

            # Write summary information to file
            with open(sat_output_txt, "w") as f:
                f.write(f"Satellite: {sat_name}, {sat}\n")
                f.write(f"Date: {date_str}\n")
                f.write(f"Nb obs.(blocks) with at least one channel with NaN in the vars of interest: {n_invalid_obs}\n")
                f.write(f"Nb total obs.(blocks) per day: {len(obs_with_invalid)}\n")
                f.write(f"Nb obs.(blocks) with NaNs per channel:\n")
                for ch, count in counts_by_channel.sort_index().items():
                    f.write(f"  Channel {ch}: {count}\n")
            #Here we refer to an observation as a block of 5 rows, each row corresponding to a channel.
            # Sometimes a block of 5 rows has NaN in different channels not just one channel.
            # For this reason the sum of the counts_by_channel does not equal to n_invalid_obs.
            print(f"Saved summary: {sat_output_txt}")

        # We specify the number of channels based on the satellite
        # For METOP-B we only use the first 4 channels
        n_channels = 4 if sat == 0 else 5
        if sat == 0:  # METOP-B, we delete the 5th channel (15)
            sat_data = sat_data[sat_data['vertco_reference_1@body'].isin([1, 2, 3, 4])]

        #We filter the blocks of rows that have NaN in any of the channels
        sat_data = filter_valid_blocks(sat_data, n_channels)

        # === 1D vars only save values from the first channel (they have repeated values for all the channels) ===
        first_chan = sat_data[sat_data['vertco_reference_1@body'] == 1]
        for out_var in target_one_d_vars:
            in_var = [k for k, v in var_mapping.items() if v == out_var][0]
            values = first_chan[in_var].values
            ds = xr.Dataset({out_var: (('obs',), values)})
            filename = os.path.join(output_dir, f"{sat_name}_{date_str}_{out_var}.nc")
            ds.coords['obs'] = ('obs', first_chan['datetime'].values)
            ds.to_netcdf(filename)

        # === Vars by channel ===
        channels = sorted(sat_data['vertco_reference_1@body'].unique())
        if sat == 0:
            channels = [ch for ch in channels if ch != 15]
        for out_var in target_chan_vars:
            in_var = [k for k, v in var_mapping.items() if v == out_var][0]
            data_by_channel = []
            channel_labels = []

            for ch in sorted(channels):
                subset = sat_data[sat_data['vertco_reference_1@body'] == ch]
                values = subset[in_var].values
                data_by_channel.append(values)
                channel_labels.append(ch)

            # convert a numpy array with the shape (channel,obs) and then transporse to (obs, channel)
            data_array = np.stack(data_by_channel, axis=0).T  # Transpose to (obs, channel)

            ds = xr.Dataset({
                out_var: (('obs', 'channel'), data_array)
            })
            ds.coords['channel'] = ('channel', channel_labels)
            ds.coords['obs'] = ('obs', first_chan['datetime'].values)
            filename = os.path.join(output_dir, f"{sat_name}_{date_str}_{out_var}.nc")
            ds.to_netcdf(filename)
            print(f"Saved: {filename}")
    print(f"[{os.getpid()}] Saved daily NetCDFs for {os.path.basename(odbFile_name)}")

# -----------------------------
# 3. Parallel Execution
# -----------------------------
def wrapper(args):
    return process_odb_file_daily(*args)

if __name__ == "__main__":
    file_args = [(f, output_dir, grid_coords_rad,export_vars,skip_txt) for f in filtered_files]


    with Pool(processes=4) as pool:
        pool.map(wrapper, file_args)
    print("All files processed.")
    print("Daily NetCDF files created in:", output_dir)
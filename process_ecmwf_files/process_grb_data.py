import sys
import os
#import earthkit.plots.quickmap
import numpy as np
import pyodc as odc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
#import earthkit.data
#import earthkit.maps
import matplotlib.pyplot as plt
import sys
import xarray as xr
#import earthkit.plots
#matplotlib.use("TkAgg")  
from datetime import datetime

# === PARAMETERS ===
filtering_land=True #If True we remove points that are not used by the satellite else False
instrument_for_training="METOP-B" #It can be "METOP-B, "METOP-C", "both"
base_path = "/perm/dnk8355/paper2026/grib_files_NH_SH/"
start_date = datetime(2024, 4, 1)
end_date = datetime(2026, 3, 31)
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# === STORAGE FOR DAILY DATA ===
skt_list = []
siconc_list_by_month = {}

# === Open the igrid file after ballTree to filter the land points in the initial conditions ===
ice_path = '/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/METOP-B_'
INITIAL_IGRID_METOPB = xr.open_dataset(ice_path + 'INITIAL_IGRID.nc')

ice_path = '/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/METOP-C_'
INITIAL_IGRID_METOPC = xr.open_dataset(ice_path + 'INITIAL_IGRID.nc')

used_indices_METOPB = INITIAL_IGRID_METOPB.INITIAL_IGRID.values
unique_METOPB = np.unique(used_indices_METOPB).astype(int)
used_indices_METOPC = INITIAL_IGRID_METOPC.INITIAL_IGRID.values
unique_METOPC = np.unique(used_indices_METOPC).astype(int)
unique_indices_METOPBandC = np.unique(np.concatenate([unique_METOPB, unique_METOPC])).astype(int)

date_list=[]
# === LOOP OVER EACH DAY ===
for date in dates:
    date_list.append(date)
    # Construct file name for each date
    file_name = f"HRES_SIC_TSKIN_N80_{date.strftime('%Y%m%d')}_NH_SH.grb"
    file_path = os.path.join(base_path, file_name)

    # Load the GRIB file using cfgrib engine
    ds = xr.open_dataset(file_path, engine="cfgrib",decode_timedelta=False)

    # We filter the dataset to keep only latitudes above 44°N and below 44°S
    # to be consistent with the fixed grid we used
    # "/perm/dnk8355/paper2026/grib_files_NH_SH/lat_lon_corrected_ref_above&below44degrees.nc"

    ds = ds.where((ds.latitude > 44) | (ds.latitude < -44), drop=True)

    # --- Extract land-sea mask (lsm) and filter for ocean points ---
    #ds = ds.where(ds.lsm == 0, drop=True)


    # --- Extract time dimension and average over time for skin temp ---
    skt = ds['skt'].mean(dim="time")        # shape: (values,)
    # --- Keep all time steps for siconc (do NOT average over time here) ---
    siconc = ds['siconc']  # shape: (time, values)

    # Keep per-day SKT
    skt_list.append(skt)

    # Group SICONC by month keeping full time dimension
    month_key = date.strftime("%Y-%m")
    if month_key not in siconc_list_by_month:
        siconc_list_by_month[month_key] = []
    siconc_list_by_month[month_key].append(siconc)

    ds.close()

# === POSTPROCESSING ===

# --- Process SKT: keep daily values (no lag) ---
skt_concat = xr.concat(skt_list, dim="DAY")  # shape: (DAY, pos)

if filtering_land==True:
    # Instead of filtering land with the lsm included in the grib file 
    # we filter the land with the filtering in the odb
    # Filter SKT to only include points from the fixed grid where there are satellite observations
    if instrument_for_training=="METOP-B":
        skt_filtered = skt_concat.isel(values=unique_METOPB)
    elif instrument_for_training=="METOP-C":
        skt_filtered = skt_concat.isel(values=unique_METOPC)
    elif instrument_for_training=="both":
        skt_filtered = skt_concat.isel(values=unique_indices_METOPBandC)
else:
    skt_filtered = skt_concat

# Extract coordinates
lon = skt_filtered.longitude.data
lat = skt_filtered.latitude.data
day_index = np.arange(len(skt_list))
date = pd.to_datetime(date_list).floor("D")

# Create a 2D array for SKT with shape (LON, DAY)
ds_skt = xr.Dataset(
    {
        "TSFC": (("LON", "DAY"), skt_filtered.values.T),  # Transpose to (LON, DAY)
        "LAT": (("LON",), lat)
    },
    coords={
        "LON": lon,
        "DAY": day_index,
        "DATE": date,
    }
)

if instrument_for_training=="METOP-B":
    ds_skt.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_tsfc_METOP-B_1apr2024_31march2026_dailyx_without_land.nc")
elif instrument_for_training=="METOP-C":
    ds_skt.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_tsfc_METOP-C_1apr2024_31march2026_dailyx_without_land.nc")
elif instrument_for_training=="both":
    ds_skt.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_tsfc_METOP-B_&_METOP-C_1apr2024_31march2026_dailyx_without_land.nc")
    
#ds_skt.to_netcdf("/perm/dnk8355/netcdf_monthly_feb2025/ifs_tsfc_feb25_dailyx_without_land.nc")
#ds_skt.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_tsfc_METOP-B_&_METOP-C_1apr2024_31march2026_dailyx_without_land.nc")

# --- Process SICONC: monthly mean, broadcasted to all days ---
day_counter = 0
siconc_daily_list = []
for month_key, daily_siconc_list in siconc_list_by_month.items():
    
    # Concatenate all days of the month into one DataArray
    all_days_in_month = xr.concat(daily_siconc_list, dim="time")  # keep all time steps

    # Compute mean of this month's siconc
    monthly_mean = all_days_in_month.mean(dim="time")  # shape: (pos,)

    # Repeat monthly mean for number of days in the month
    n_days = len(daily_siconc_list)
    day_range = np.arange(day_counter, day_counter + n_days)
    monthly_broadcast = (monthly_mean.expand_dims(DAY=n_days).assign_coords(DAY=day_range))

    siconc_daily_list.append(monthly_broadcast)
    day_counter += n_days

# Concatenate all months
siconc_all = xr.concat(siconc_daily_list, dim="DAY")  # shape: (DAY, pos)
# Add the corresponding date for each DAY
siconc_all = siconc_all.assign_coords(DATE=("DAY", dates))

# --- Add lag: duplicate first day of siconc ---
# duplicate first day
first_day = siconc_all.isel(DAY=0).expand_dims(DAY=[0])

# shift the original days by 1
siconc_all_shifted = siconc_all.assign_coords(DAY=siconc_all.DAY + 1)

# Concatenate lag day + original series
siconc_with_lag = xr.concat([first_day, siconc_all_shifted], dim="DAY")
# Create DATE coordinate including the lag day
lag_date = dates[0] - pd.Timedelta(days=1)
dates_with_lag = pd.DatetimeIndex([lag_date]).append(dates)
# Assign DATE coordinate
siconc_with_lag = siconc_with_lag.assign_coords(
    DATE=("DAY", dates_with_lag)
)

if filtering_land==True:
    # Filter sic to only include points from the fixed grid where there are satellite observations
    if instrument_for_training=="METOP-B":
        siconc_with_lag_filtered = siconc_with_lag.isel(values=unique_METOPB)
    elif instrument_for_training=="METOP-C":
        siconc_with_lag_filtered = siconc_with_lag.isel(values=unique_METOPC)
    elif instrument_for_training=="both":
        siconc_with_lag_filtered = siconc_with_lag.isel(values=unique_indices_METOPBandC)
else: 
    siconc_with_lag_filtered = siconc_with_lag


# Fill NaNs (which represents land in land sea mask of the grib file) with 0
siconc_with_lag_filtered=siconc_with_lag_filtered.fillna(0)
# Convert longitudes from [0, 360] to [-180, 180]
#siconc_with_lag_recentered = siconc_with_lag.assign_coords(
#    longitude=(((siconc_with_lag.longitude + 180) % 360) - 180)
#)

#Reorder by longitude after recentering 
#siconc_with_lag_recentered = siconc_with_lag_recentered.sortby("longitude")

seaice_data = siconc_with_lag_filtered.values.T  # now shape: (pos=LON, DAY)

ds_siconc = xr.Dataset(
    {
        "SEAICE": (("LON", "DAY"), seaice_data),
        "LAT": (("LON",), siconc_with_lag_filtered.latitude.data)
    },
    coords={
        "LON": siconc_with_lag_filtered.longitude.data,
        "DAY": siconc_with_lag_filtered.DAY
    }
)

if instrument_for_training=="METOP-B":
    ds_siconc.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_seaice_initials_METOP-B_1apr2024_31march2025_without_land_without_nans.nc")
elif instrument_for_training=="METOP-C":
    ds_siconc.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_seaice_initials_METOP-C_1apr2024_31march2025_without_land_without_nans.nc")
elif instrument_for_training=="both":
    ds_siconc.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_seaice_initials_METOP-B_&_METOP-C_1apr2024_31march2025_without_land_without_nans.nc")
#ds_siconc.to_netcdf("/perm/dnk8355/netcdf_monthly_feb2025/ifs_seaice_initials_METOP-B_feb2025_without_land_without_nans.nc")
#ds_siconc.to_netcdf("/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/ifs_seaice_initials_METOP-B_&_METOP-C_1apr2024_31march2025_without_land_without_nans.nc")


#plt.scatter(ds_skt.LON,ds_skt.LAT,c=ds_skt.TSFC[:,0])
#plt.scatter(ds_siconc.LON,ds_siconc.LAT,c=ds_siconc.SEAICE[:,0])
#ds_siconc = xr.Dataset({"SEAICE": siconc_with_lag.transpose("values", "DAY")})
#Path to the file
#file_path_sic = "/perm/dnk8355/netcdf_monthly_feb2025/ifs_seaice_initials_feb25.nc"
#file_path_skt = "/perm/dnk8355/netcdf_monthly_feb2025/ifs_tsfc_feb25_dailyx.nc"

#file_path_sic = "/home/dnk8355/myPerm/sea_ice_data/ifs_seaice_initials_year.nc"
#file_path_skt = "/home/dnk8355/myPerm/sea_ice_data/ifs_tsfc_year_dailyx.nc"

#file_pathgrid = "/home/dnk8355/myPerm/sea_ice_data/field_v2_IGRID.nc"
#file_mypathgrid = "/home/dnk8355/myPerm/netcdf_monthly_may2024/METOP-B_202405_IGRID.nc"

# Load the dataset
#sic = xr.open_dataset(file_path_sic)
#skt = xr.open_dataset(file_path_skt)
#df_grid = xr.open_dataset(file_pathgrid)
#df_grid.IGRID.data.max()+1
#df_mygrid = xr.open_dataset(file_mypathgrid)


#Path of my grib files

#file_name="/perm/dnk8355/grib_files_NH/HRES_SIC_TSKIN_N80_20240314_NH_variables.grb"
# We load the grib file in a xarray dataset
#ds = xr.open_dataset(file_name, engine="cfgrib")

# Extract variables
#lons = ds.longitude
#lats = ds.latitude
#siconc = ds.siconc  # Sea ice concentration

#ds_filtered = ds.where((ds.lsm < 0.01) & (ds.latitude >= 50), drop=True)


# Convert longitudes from [0, 360] to [-180, 180]
#skt_concat = skt_concat.assign_coords(
#    longitude=((skt_concat.longitude + 180) % 360 - 180)
#)

#Reorder by longitude after recentering 
#skt_recentered = skt_recentered.sortby("longitude")

# We convert to dataframe
#skt_df = skt_concat.to_dataframe().reset_index()


#skt_concat = skt_concat.assign_coords(
#    longitude=((skt_concat.longitude + 180) % 360 - 180)
#)
#skt_df = skt_concat.to_dataframe().reset_index()
#grid_df = reduced_grid[['lat','lon']].to_dataframe().reset_index()
#merged = pd.merge(
#    skt_df,
#    grid_df,
#    left_on=['latitude','longitude'],
#    right_on=['lat','lon']
#)
#matching_idx = merged['values'].unique()
#skt_filtered = skt_concat.isel(values=matching_idx)

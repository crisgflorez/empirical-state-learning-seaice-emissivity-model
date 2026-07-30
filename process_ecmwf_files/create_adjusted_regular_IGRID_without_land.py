import sys
import os
import numpy as np
import pyodc as odc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import sys
import xarray as xr  
from datetime import datetime
import glob

# === Load full grid ===
grid_ds = xr.open_dataset("/home/dnk8355/perm/paper2026/grib_files_NH_SH/lat_lon_corrected_ref_above&below44degrees.nc")
lons_ref = grid_ds.lon.values
lats_ref = grid_ds.lat.values

# Create IGRID.nc for running experiments using only one satellite
# and their associated reduced fixed grid
sensors=['METOP-B','METOP-C']
for i in sensors:
    # === Load INITIAL_IGRID ===
    ice_path = '/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/' + str(i)
    INITIAL_IGRID = xr.open_dataset(ice_path + '_INITIAL_IGRID.nc')
    old_indices = INITIAL_IGRID.INITIAL_IGRID.values

    # === Get unique used indices ===
    used_indices = np.unique(old_indices).astype(int)

    # === Create mapping from old igrid -> new igrid ===
    mapping = {old: new for new, old in enumerate(used_indices)}

    # === Apply mapping to remap all observation indices ===
    new_indices = np.array([mapping[idx] for idx in old_indices])

    # === Create reduced grid dataset ===

    reduced_grid_ds = xr.Dataset(
        {
            "lat": (("igrid",), lats_ref[used_indices]),
            "lon": (("igrid",), lons_ref[used_indices]),
        },
        coords={"igrid": np.arange(len(lats_ref[used_indices]))}
    )
    # === Create new IGRID dataset ===
    # This IGRID do not consider land points
    IGRID = xr.Dataset(
        {"IGRID": (("OBS",), new_indices)}
    )

    # === Save everything to NetCDF ===

    reduced_grid_ds.to_netcdf('/home/dnk8355/perm/paper2026/grib_files_NH_SH/'+ str(i)+'_1april2024_31march2026_lat_lon_corrected_ref_above44_without_land.nc')
    IGRID.to_netcdf('/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/'+ str(i)+'_IGRID.nc')


# Create IGRID.nc for running experiments using METOP-B and METOP-C together
# and their associated reduced fixed grid
# === Load INITIAL_IGRID for each satellite ===
ice_path = '/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/METOP-B_'
INITIAL_IGRID_METOPB = xr.open_dataset(ice_path + 'INITIAL_IGRID.nc')

ice_path = '/perm/dnk8355/paper2026/netcdf_1april2024_31march2026/METOP-C_'
INITIAL_IGRID_METOPC = xr.open_dataset(ice_path + 'INITIAL_IGRID.nc')

used_indices_METOPB = INITIAL_IGRID_METOPB.INITIAL_IGRID.values
unique_METOPB = np.unique(used_indices_METOPB).astype(int)
used_indices_METOPC = INITIAL_IGRID_METOPC.INITIAL_IGRID.values
unique_METOPC = np.unique(used_indices_METOPC).astype(int)

# Used indices taking into account METOP-B and METOP-C
used_indices = np.unique(np.concatenate([unique_METOPB, unique_METOPC])).astype(int)   

# === Create mapping from old igrid -> new igrid ===
mapping = {old: new for new, old in enumerate(used_indices)}

# === Apply mapping to remap all observation indices in METOP-B ===
new_indices_METOPB = np.array([mapping[idx] for idx in used_indices_METOPB])
# === Apply mapping to remap all observation indices in METOP-C ===
new_indices_METOPC = np.array([mapping[idx] for idx in used_indices_METOPC])


# === Create reduced grid dataset ===

reduced_grid_ds = xr.Dataset(
    {
        "lat": (("common_igrid",), lats_ref[used_indices]),
        "lon": (("common_igrid",), lons_ref[used_indices]),
    },
    coords={"igrid": np.arange(len(lats_ref[used_indices]))}
)

# === Create new IGRID dataset both for METOPB and METOPC 
# for training together===
# This IGRID do not consider land points
IGRID_common_METOPB = xr.Dataset(
    {"COMMON_IGRID": (("OBS",), new_indices_METOPB)}
)
IGRID_common_METOPC = xr.Dataset(
    {"COMMON_IGRID": (("OBS",), new_indices_METOPC)}
)


# === Save everything to NetCDF ===
reduced_grid_ds.to_netcdf('/home/dnk8355/perm/paper2026/grib_files_NH_SH/'+ str(i)+'_1april2024_31march2026_lat_lon_corrected_ref_above44_without_land.nc')
IGRID_common_METOPB.to_netcdf('/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/'+'METOP-B_COMMON_IGRID.nc')
IGRID_common_METOPC.to_netcdf('/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/'+'METOP-C_COMMON_IGRID.nc')



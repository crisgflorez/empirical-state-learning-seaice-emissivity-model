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


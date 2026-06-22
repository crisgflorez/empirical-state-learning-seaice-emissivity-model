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
from sklearn.neighbors import BallTree

#Path of my grib file
file_name="/perm/dnk8355/paper2026/grib_files_NH_SH/HRES_SIC_TSKIN_N80_20240401_NH_SH.grb"
# We load the grib file in a xarray dataset
ds = xr.open_dataset(file_name, engine="cfgrib")
# We filter the dataset to keep only latitudes above 45°N and below 45°S
ds = ds.where((ds.latitude > 44) | (ds.latitude < -44), drop=True)
# Extract all latitudes and longitudes from N80 reduced gaussian grid
lat_all = ds.latitude.values
lon_all = ds.longitude.values

# Convert longitudes from [0, 360] to [-180, 180]
#Save lat and lon corrected into a netcdf file
lon_corrected = (lon_all + 180) % 360 - 180

# Path to save NetCDF
output_file = "/perm/dnk8355/paper2026/grib_files_NH_SH/lat_lon_corrected_ref_above&below44degrees.nc"

# Create Dataset
ds_out = xr.Dataset(
    {
        "lat": (("points",), lat_all),
        "lon": (("points",), lon_corrected),
    }
)

# Save to NetCDF
ds_out.to_netcdf(output_file)

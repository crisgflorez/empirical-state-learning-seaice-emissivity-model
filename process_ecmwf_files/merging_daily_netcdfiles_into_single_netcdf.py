import xarray as xr
import glob
import os
from itertools import product
import pandas as pd

#With this script daily Netcdf files are merged into monthly or full period Netcdf files

# Satellite names and variables
sats = ['METOP-B', 'METOP-C']
#names_nc = ['FG_DEP']
names_nc = [
    'LAT','LON','JULIAN_DAY','INITIAL_IGRID',
    'OBSVALUE','TSFC','WINDSPEED10M','CLOUD_FRACTION','EMIS_WATER',
    'TAUSFC','TDOWN','TUP','TAUSFC_CLD',
    'TDOWN_CLD','TUP_CLD','NEAREST_LATS','NEAREST_LONS','SEAICE','ZENITH',
    'AZIMUTH','SCANLINE', 'SCANPOS','FG_DEP','AN_DEP'
]


daily_dir = '/perm/dnk8355/paper2026/netcdf_daily_1april2024_31march2026'
monthly_dir = '/perm/dnk8355/paper2026/netcdf_1april2024_31march2026'
os.makedirs(monthly_dir, exist_ok=True)

pairs = set(product(sats, names_nc))

for sat, var in sorted(pairs):
    pattern = os.path.join(daily_dir, f"{sat}_*_{var}.nc")
    matching_files = sorted(glob.glob(pattern))

    if not matching_files:
        continue

    #File 01/04/2024 has obs values from 31/03/2024, so we need to remove obs values before 01/04/2024 from this file
    first_file = matching_files[0]

    date_str = os.path.basename(first_file).split("_")[1]
    day_start = pd.to_datetime(date_str, format="%Y%m%d")

    tmp_file = first_file.replace(".nc", "_tmp.nc")

    with xr.open_dataset(first_file) as ds:
        ds = ds.where(ds["obs"] >= day_start, drop=True)
        ds.to_netcdf(tmp_file)

    os.replace(tmp_file, first_file)

    print(f"Merging {len(matching_files)} files for {sat} - {var}")

    try:
        ds = xr.open_mfdataset(matching_files, concat_dim='obs', combine='nested')
        monthly_file = os.path.join(monthly_dir, f"{sat}_{var}.nc")
        ds.to_netcdf(monthly_file)
        print(f"Saved: {monthly_file}")

        # Delete daily files after successful merge
        #for f in matching_files:
        #    os.remove(f)

    except Exception as e:
        print(f" Error merging {sat} - {var}: {e}")

print(" All NetCDF files saved")
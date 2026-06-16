import xarray as xr
import glob
import os
from itertools import product

#With this scriot daily Netcdf files are merged into monthly or full period Netcdf files

# Satellite names and variables
sats = ['METOP-B', 'METOP-C', 'NOAA-15', 'NOAA-18', 'NOAA-19']
names_nc = ['FG_DEP']
#names_nc = [
#    'LAT','LON','JULIAN_DAY','INITIAL_IGRID',
#    'OBSVALUE','TSFC','WINDSPEED10M','CLOUD_FRACTION','EMIS_WATER',
#    'TAUSFC','TDOWN','TUP','TAUSFC_CLD',
#    'TDOWN_CLD','TUP_CLD','NEAREST_LATS','NEAREST_LONS','SEAICE','ZENITH',
#    'AZIMUTH','SCANLINE', 'SCANPOS'
#]


daily_dir = '/perm/dnk8355/netcdf_daily_april2024_april2025'
monthly_dir = '/perm/dnk8355/netcdf_1april2024_31march2025'
os.makedirs(monthly_dir, exist_ok=True)

pairs = set(product(sats, names_nc))

for sat, var in sorted(pairs):
    pattern = os.path.join(daily_dir, f"{sat}_*_{var}.nc")
    matching_files = sorted(glob.glob(pattern))

    if not matching_files:
        continue

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
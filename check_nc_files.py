import os
import glob
import xarray as xr
import pandas as pd
from multiprocessing import Pool
import numpy as np

# =========================================================
# 1. PATH
# =========================================================
data_dir = "/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026"

# =========================================================
# 2. CONFIG: LAT FILTER (None = no filter)
# =========================================================
lat_limit = 50   # <- None if we don't want to filter

# =========================================================
# 3. LOAD LAT FILES 
# =========================================================
lat_files = glob.glob(os.path.join(data_dir, "*LAT.nc"))

lat_dict = {}

for lf in lat_files:
    sat = os.path.basename(lf).split("_")[0]

    with xr.open_dataset(lf) as ds_lat:
        lat_dict[sat] = ds_lat["LAT"].values

# =========================================================
# 4. FIND ALL NETCDF FILES
# =========================================================
nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
vars_of_interest = [
    'TSFC','WINDSPEED10M','CLOUD_FRACTION','SEAICE','OBSVALUE',
    'EMIS_WATER','TAUSFC','TDOWN','TUP','TAUSFC_CLD',
    'TDOWN_CLD','TUP_CLD','FG_DEP','AN_DEP'
]

nc_files = [
    f for f in nc_files
    if any(v in os.path.basename(f) for v in vars_of_interest)
]

# =========================================================
# 5. WORKER FUNCTION 
# =========================================================
def process_file(f, lat_dict, lat_limit):

    sat = os.path.basename(f).split("_")[0]

    with xr.open_dataset(f) as ds:

        # ---------------------------------
        # Time (always full first)
        # ---------------------------------
        time_obs = pd.to_datetime(ds["obs"].values)

        # ---------------------------------
        # Hemisphere (always global definition)
        # ---------------------------------
        lats = lat_dict[sat]
        hemisphere = np.where(lats > 0, "North", "South")

        # ---------------------------------
        # Optional latitude filter
        # ---------------------------------
        if lat_limit is not None:

            mask = np.abs(lats) >= lat_limit
            time_obs = time_obs[mask]
            hemisphere = hemisphere[mask]

        records_daily_all = []
        records_daily_hemis = []

        # =============================================
        # Loop over variables
        # =============================================
        for var in ds.data_vars:

            da = ds[var]

            # =============================
            # Variables with channels
            # =============================
            if "channel" in da.dims:

                for ch in ds["channel"].values:

                    values = da.sel(channel=ch).values
                    if lat_limit is not None:
                        values = values[mask]                    

                    df = pd.DataFrame({
                        "day": time_obs.date,
                        "hemisphere": hemisphere,
                        "values": values
                    })

                    daily_all = (
                        df.groupby(["day", "hemisphere"])["values"]
                          .agg(["min", "max", "mean", "count"])
                          .reset_index()
                    )

                    daily_all["file"] = os.path.basename(f)
                    daily_all["var"] = var
                    daily_all["channel"] = ch
                    daily_all["satellite"] = sat

                    daily_hemis = (
                        df.groupby(["day"])["values"]
                          .agg(["min", "max", "mean", "count"])
                          .reset_index()
                    )

                    daily_hemis["file"] = os.path.basename(f)
                    daily_hemis["var"] = var
                    daily_hemis["channel"] = ch
                    daily_hemis["satellite"] = sat                    

                    records_daily_all.append(daily_all)
                    records_daily_hemis.append(daily_hemis)
                    
            # =============================
            # Variables without channels
            # =============================
            else:

                values = da.values

                if lat_limit is not None:
                    values = values[mask]                

                df = pd.DataFrame({
                    "day": time_obs.date,
                    "hemisphere": hemisphere,
                    "values": values
                })

                daily_all = (
                    df.groupby(["day", "hemisphere"])["values"]
                      .agg(["min", "max", "mean", "count"])
                      .reset_index()
                )

                daily_all["file"] = os.path.basename(f)
                daily_all["var"] = var
                daily_all["channel"] = pd.NA
                daily_all["satellite"] = sat

                daily_hemis = (
                    df.groupby(["day"])["values"]
                        .agg(["min", "max", "mean", "count"])
                        .reset_index()
                )

                daily_hemis["file"] = os.path.basename(f)
                daily_hemis["var"] = var
                daily_hemis["channel"] = pd.NA
                daily_hemis["satellite"] = sat                    

                records_daily_all.append(daily_all)
                records_daily_hemis.append(daily_hemis)

    return {
        "daily_hem": pd.concat(records_daily_all, ignore_index=True),
        "daily_global": pd.concat(records_daily_hemis, ignore_index=True)
    }


# =========================================================
# 6. PARALLEL EXECUTION
# =========================================================
if __name__ == "__main__":

    nproc = 4  # adjust
    args = [(f, lat_dict, lat_limit) for f in nc_files]
    
    with Pool(processes=nproc) as pool:
        results = pool.starmap(process_file, args)

    # =====================================================
    # 6.1 MERGE FINAL RESULTS
    # =====================================================

    df_hem = pd.concat(
        [r["daily_hem"] for r in results],
        ignore_index=True
    )

    df_global = pd.concat(
        [r["daily_global"] for r in results],
        ignore_index=True
    )

    # =====================================================
    # 6.2 SAVE
    # =====================================================

    out_hem = os.path.join(
        data_dir,
        "daily_by_hemisphere_above_below_50.pkl"
    )

    out_global = os.path.join(
        data_dir,
        "daily_global_above_below_50.pkl"
    )

    df_hem.to_pickle(out_hem)
    df_global.to_pickle(out_global)

    print("Saved hemisphere:", out_hem)
    print("Saved global:", out_global)

    print("Rows hem:", len(df_hem))
    print("Rows global:", len(df_global))

    
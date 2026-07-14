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
from datetime import datetime, timedelta
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

folder = "/home/dnk8355/perm/paper2026/outputs_training/exp1_METOPB"
#folder = "/home/dnk8355/perm/paper2026/outputs_training/exp1_METOPC_obs_err_fromMETOPB"

tag_name='8epochs'


#scanpos - scan positions for each observation
scanpos=xr.open_dataset("/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/METOP-B_SCANPOS.nc")
#scanpos=xr.open_dataset("/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/METOP-C_SCANPOS.nc")

#FG_dep for each observation
fg_dep=xr.open_dataset("/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/METOP-B_FG_DEP.nc")
#fg_dep=xr.open_dataset("/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/METOP-C_FG_DEP.nc")

# Collect all .nc files in the folder
files = sorted(glob.glob(os.path.join(folder, "*_8epochs*.nc")))

datasets = [xr.open_dataset(f) for f in files]


#tbobs - observed brightness temperatures
tbobs=datasets[5]

#tbsim - simulated brightness temperatures from the model
tbsim=datasets[6]

#tbsim initial - initial simulated brightness temperatures from the model
tbsim_ini=datasets[7]


# Create masks for specific scan positions
mask_scanpos_14_15_16 = scanpos['SCANPOS'].isin([14, 15, 16])
mask_scanpos_edge = (
    ((scanpos['SCANPOS'] >= 0) & (scanpos['SCANPOS'] <= 5)) |
    ((scanpos['SCANPOS'] >= 25) & (scanpos['SCANPOS'] <= 30))
)

fg_dep_scanpos_14_15_16 = fg_dep.where(mask_scanpos_14_15_16, drop=True)
fg_dep_scanpos_edge = fg_dep.where(mask_scanpos_edge, drop=True)

#We rename the dimension 'obs' to 'iobs' to match the dimension name in tbobs
mask_scanpos_14_15_16 = mask_scanpos_14_15_16.rename(obs="iobs")
mask_scanpos_14_15_16 = mask_scanpos_14_15_16.assign_coords(iobs=tbobs.iobs)
mask_scanpos_edge = mask_scanpos_edge.rename(obs="iobs")
mask_scanpos_edge = mask_scanpos_edge.assign_coords(iobs=tbobs.iobs)

tbobs_scanpos_14_15_16 = tbobs.where(mask_scanpos_14_15_16, drop=True)
tbsim_scanpos_14_15_16 = tbsim.where(mask_scanpos_14_15_16, drop=True)

tbobs_scanpos_edge = tbobs.where(mask_scanpos_edge, drop=True)
tbsim_scanpos_edge = tbsim.where(mask_scanpos_edge, drop=True)


mean_fg_dep_per_channel = fg_dep['FG_DEP'].mean(dim="obs")

diff_all = tbobs['tb'] - tbsim['tb']
mean_diff_per_channel = diff_all.mean(dim="iobs")
#STD
std_diff_per_channel = diff_all.std(dim="iobs")
# MSE
mse_per_channel = (diff_all ** 2).mean(dim="iobs")
# RMSE 
rmse_per_channel = np.sqrt(mse_per_channel)


mean_fg_dep_per_channel_scanpos_14_15_16 = fg_dep_scanpos_14_15_16['FG_DEP'].mean(dim="obs")

diff_scanpos_14_15_16 = tbobs_scanpos_14_15_16['tb'] - tbsim_scanpos_14_15_16['tb']
mean_diff_per_channel_scanpos_14_15_16 = diff_scanpos_14_15_16.mean(dim="iobs")
std_diff_per_channel_scanpos_14_15_16 = diff_scanpos_14_15_16.std(dim="iobs")
mse_per_channel_scanpos_14_15_16 = (diff_scanpos_14_15_16 ** 2).mean(dim="iobs")
rmse_per_channel_scanpos_14_15_16 = np.sqrt(mse_per_channel_scanpos_14_15_16)


mean_fg_dep_per_channel_edge = fg_dep_scanpos_edge['FG_DEP'].mean(dim="obs")

diff_scanpos_edge = tbobs_scanpos_edge['tb'] - tbsim_scanpos_edge['tb']     
mean_diff_per_channel_edge = diff_scanpos_edge.mean(dim="iobs")
std_diff_per_channel_edge = diff_scanpos_edge.std(dim="iobs")
mse_per_channel_edge = (diff_scanpos_edge ** 2).mean(dim="iobs")
rmse_per_channel_edge = np.sqrt(mse_per_channel_edge)


#Table with statistics
channels = tbobs.channel_name.values

# Crear la tabla
results = pd.DataFrame(
    index=[
        "Mean FG dep (all)",
        "Mean FG dep (scanpos 14-16)",
        "Mean FG dep (edge)",
        "Mean (all)",
        "Mean (scanpos 14-16)",
        "Mean (edge)",
        "STD (all)",
        "STD (scanpos 14-16)",
        "STD (edge)",
        "RMSE (all)",
        "RMSE (scanpos 14-16)",
        "RMSE (edge)"
    ],
    columns=channels,
    data= [ mean_fg_dep_per_channel.values,
        mean_fg_dep_per_channel_scanpos_14_15_16.values,
        mean_diff_per_channel_edge.values,
        mean_fg_dep_per_channel_edge.values,
        mean_diff_per_channel_scanpos_14_15_16.values,
        mean_diff_per_channel_edge.values,
        std_diff_per_channel.values,
        std_diff_per_channel_scanpos_14_15_16.values,
        std_diff_per_channel_edge.values,
        rmse_per_channel.values,
        rmse_per_channel_scanpos_14_15_16.values,
        rmse_per_channel_edge.values]
)

print(results)
results.to_csv(f"{folder}/statistics_{tag_name}.csv")

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
import string
from matplotlib.ticker import AutoMinorLocator, NullLocator
from string import ascii_lowercase


# Load experiment data
def load_experiment(folder, tag):
    files = sorted(glob.glob(os.path.join(folder, f"*{tag}*.nc")))
    print(list(files))
    datasets = [xr.open_dataset(f) for f in files]

    return {
        "ice_emis": datasets[0],
        "models": datasets[1],
        "properties_grid": datasets[2],
        "properties_obs": datasets[3],
        "seaice": datasets[4],
        "tbobs": datasets[5],
        "tbsim": datasets[6],
        "tbsim_ini": datasets[7],
        "file_names":files

    }


# Filter one day
def filter_by_day(tbobs, tbsim, tbsim_ini, fg_dep, scanpos, date_tb, date_odb):

    tbobs_f = tbobs.where(
        tbobs["date_time_fromjd"].dt.floor("D") == date_tb, drop=True
    )
    tbsim_f = tbsim.where(
        tbsim["date_time_fromjd"].dt.floor("D") == date_tb, drop=True
    )
    tbsim_ini_f = tbsim_ini.where(
        tbsim_ini["date_time_fromjd"].dt.floor("D") == date_tb, drop=True
    )

    fg_dep_f = fg_dep.where(
        fg_dep.obs.dt.floor("D") == date_odb, drop=True
    )

    scanpos_f = scanpos.where(
        fg_dep.obs.dt.floor("D") == date_odb, drop=True
    )
    
    return tbobs_f, tbsim_f, tbsim_ini_f, fg_dep_f, scanpos_f



folder = "/perm/dnk8355/outputs_training_v2_jan26_report_final"
folder_plot = "/perm/dnk8355/outputs_training_v2_jan26_report_final/fellow_day"
experiments = {
    "exp1": "bg_emis08_with_losses_original_obs_errors_bg_biasice2_5_ocean5_bg_bias_err0_001_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10",
    "exp2": "bg_emis08_with_losses_new_obs_errors_bg_biasice2_5_ocean5_bg_bias_err0_001_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10",
    "exp3": "bg_emis07_with_losses_new_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_02_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10",
    "exp4": "bg_emis07_with_losses_new_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_02_newimplementation_in_emisNN_with_angle_sbatch_19jan_python3_10",
    "exp5": "bg_emis06_with_losses_new_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_02_newimplementation_in_emisNN_with_angle_sbatch_19jan_python3_10",
}
scanpos = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/METOP-B_SCANPOS.nc")
fg_dep = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/METOP-B_FG_DEP.nc")


#Plot of SIC from hybrid ML model, IFS and differences for 1 day
#For this case we only use experiment 4
exp4 = load_experiment(folder, experiments['exp4'])
seaice_exp4=exp4['seaice']
#For the moment we transform negative values in nans
seaice_exp4 = seaice_exp4.where(seaice_exp4.seaice >= 0)

def day_index(date_str, start_str="2024-04-01"):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start = datetime.strptime(start_str, "%Y-%m-%d")
    return (date - start).days 

day=day_index("2025-02-01")


# --- First dataset: analysis ---
seaice_analysis = seaice_exp4.seaice.isel(lagstep=day).values
lon_analysis = seaice_exp4.lon.isel(lagstep=day).values
lat_analysis = seaice_exp4.lat.isel(lagstep=day).values

def plot_sic_analysis(
    lon_analysis,
    lat_analysis,
    sic_analysis,
    date_str,output_path,
    lat_lims=(50, 90),
    figsize=(24, 8.5),
    point_size=20):
    """
    Plot sea-ice concentration analysis, IFS SIC, and their difference
    on polar stereographic maps with consistent colormaps and colorbars.
    """

    # ------------------------------------------------------------------
    # 2. Normalization and colormaps
    # ------------------------------------------------------------------
    sic_min = np.nanmin(sic_analysis)
    sic_max = np.nanmax(sic_analysis)

    norm_ice = Normalize(vmin=sic_min, vmax=sic_max)

    cmap_ice = LinearSegmentedColormap.from_list(
        'ice_cmap',
        [(0.0, 0.0, 0.3),
         (0.0, 0.2, 0.6),
         (0.5, 0.7, 1.0),
         (1.0, 1.0, 1.0)],
        N=100
    )

    # ------------------------------------------------------------------
    # 3. Figure and axes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.NorthPolarStereo()}
    )
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]],
                    crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='gray')
    ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
    ax.gridlines()

    latitudes = np.arange(50, 81, 10)

    for lat in latitudes:
        ax.text(
            120, lat,                     # <- aquí eliges 60°E
            f"{lat}°N",
            transform=ccrs.PlateCarree(),
            ha='left',
            va='center',
            fontsize=18
        )

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    sc1 = ax.scatter(
        lon_analysis, lat_analysis,
        c=sic_analysis,
        s=point_size,
        cmap=cmap_ice,
        norm=norm_ice,
        transform=ccrs.PlateCarree()
    )
    ax.set_title("1 Feb 2025", fontsize=18)

    # ------------------------------------------------------------------
    # 5. Colorbar
    # ------------------------------------------------------------------

    pos = ax.get_position()
    cbar_y = pos.y0 - 0.10
    cbar_h = 0.025

    cax2 = fig.add_axes([
        pos.x0,
        cbar_y,
        pos.width,
        cbar_h
    ])

    cb2 = fig.colorbar(sc1, cax=cax2, orientation='horizontal')
    cb2.set_label('Sea Ice Concentration', fontsize=22, labelpad=-65)
    cb2.ax.tick_params(labelsize=20)

    fig.savefig(f"{output_path}/SIC_01022025_exp4_without_nans"+'.png', dpi=300, bbox_inches="tight")


    return fig, ax



fig, ax = plot_sic_analysis(
    lon_analysis,
    lat_analysis,
    seaice_analysis,
    date_str="01/02/2025",output_path=folder_plot,point_size=60
)


target_date = np.datetime64("2024-04-01")
target_date_odb = np.datetime64("2024-04-01") 

#For comparison between experiments
tbsim_all = {} #We save the simulated BT from a specific day from all the experiments
for name, tag in experiments.items():    
    exp = load_experiment(folder, tag)

    tbobs_f, tbsim_f, tbsim_ini_f, fg_dep_f, scanpos_f = filter_by_day(
        exp["tbobs"],
        exp["tbsim"],
        exp["tbsim_ini"],
        fg_dep, scanpos,
        target_date,
        target_date_odb
    )
    tbsim_all[name] = tbsim_f




def plot_tb_single_experiment(
    tbobs_f,
    tbsim_all,
    exp_name,
    channel,
    title,
    outfile,
    lat_lims=(50, 90)
):

    channel_names = ["23V", "31V", "50V", "53V"]

    # --- Select data ---
    tb_obs = tbobs_f.isel(channel=channel)
    tb_sim = tbsim_all[exp_name].isel(channel=channel)

    tb_diff = tb_obs.tb - tb_sim.tb

    # --- Common limits for Obs and Sim ---
    vmin_bt = min(tb_obs.tb.min().item(), tb_sim.tb.min().item())
    vmax_bt = max(tb_obs.tb.max().item(), tb_sim.tb.max().item())

    # --- Limits for difference ---
    vmin_diff = tb_diff.min().item()
    vmax_diff = tb_diff.max().item()

    zero_pos = (0 - vmin_diff) / (vmax_diff - vmin_diff)

    cmap_diff = LinearSegmentedColormap.from_list(
        'blue_white_red',
        [(0.0, 'navy'), (zero_pos, 'white'), (1.0, 'darkred')]
    )

    # --- Figure ---
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5),
        subplot_kw={'projection': ccrs.NorthPolarStereo()}
    )

    titles = ["Observed \n Brightness Temperature", f"Simulated \n Brightness Temperature", "Obs − Sim"]
    datasets = [tb_obs.tb, tb_sim.tb, tb_diff]
    cmaps = ["jet", "jet", cmap_diff]
    norms = [
        Normalize(vmin=vmin_bt, vmax=vmax_bt),
        Normalize(vmin=vmin_bt, vmax=vmax_bt),
        Normalize(vmin=vmin_diff, vmax=vmax_diff)
    ]

    for i, ax in enumerate(axes):

        ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]],
                      crs=ccrs.PlateCarree())

        im = ax.scatter(
            tb_obs.lon,
            tb_obs.lat,
            c=datasets[i],
            s=10,
            cmap=cmaps[i],
            norm=norms[i],
            transform=ccrs.PlateCarree()
        )

        ax.set_title(titles[i], fontsize=20)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines()
        
        pos = ax.get_position()
        cbar_y = pos.y0 - 0.05
        cbar_h = 0.025

        cax2 = fig.add_axes([
            pos.x0,
            cbar_y,
            pos.width,
            cbar_h
        ])

        cbar = plt.colorbar(im, cax=cax2, orientation="horizontal",
                            shrink=0.8, pad=0.05)
        cbar.ax.tick_params(labelsize=16)


    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig, axes


plot_tb_single_experiment(
    tbobs_f,
    tbsim_all,
    exp_name="exp1",
    channel=1,
    title=str(pd.to_datetime(target_date_odb).strftime("%d/%m/%Y")),
    outfile=os.path.join(folder_plot, f"BT_single_exp_{target_date_odb}.png")
)
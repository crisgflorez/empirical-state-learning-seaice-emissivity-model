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

def load_experiment(output_path, pattern):
    """
    Load tb observations and simulations from an experiment folder.
    """
    files = sorted(glob.glob(os.path.join(output_path, pattern)))

    datasets = [xr.open_dataset(f) for f in files]

    tbobs = datasets[5]
    tbsim = datasets[6]

    diff = tbobs['tb'] - tbsim['tb']

    return tbobs, tbsim, diff


#Plot for 3 days to check channels in METOPC
def polar_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]],
                  crs=ccrs.PlateCarree())

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

def select_day_channel(tbobs, diff, target_date, channel):
    """
    Select one day and one channel.
    If the channel does not exist, return None.
    """

    mask = tbobs.date_time_fromjd.dt.floor("D") == target_date

    tbobs_day = tbobs.where(mask, drop=True)
    diff_day = diff.where(mask, drop=True)

    # Check if channel exists
    if channel >= tbobs.sizes["channel"]:
        return None, None

    tbobs_ch = tbobs_day.isel(channel=channel)
    diff_ch = diff_day.isel(channel=channel)

    return tbobs_ch, diff_ch

def select_hemisphere(tbobs, diff, hemisphere):
    """
    Select Northern or Southern Hemisphere.
    Returns None if the dataset is not available.
    """

    if tbobs is None or diff is None:
        return None, None

    if hemisphere == "NH":
        mask = tbobs.lat >= 44

    elif hemisphere == "SH":
        mask = tbobs.lat <= -44

    else:
        raise ValueError("hemisphere must be NH or SH")

    return (
        tbobs.where(mask, drop=True),
        diff.where(mask, drop=True)
    )

def plot_difference_map(ax, tbobs_day, diff, norm):

    if tbobs_day is None:
        ax.text(
            0.5,
            0.5,
            "Channel not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14
        )


        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        return None


    sc = ax.scatter(
        tbobs_day.lon,
        tbobs_day.lat,
        c=diff,
        s=15,
        cmap="RdBu_r",
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.gridlines()


    return sc

#Parameters
target_date = np.datetime64("2025-06-16")   # Select the day
channel = 4                              # Select the channel
#Channels: 23.8 GHz, 31.4 GHz, 50.3 GHz, 52.8 GHz, 89GHz
channel_name="89 GHz"

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

output_path_metopB = (
    "/home/dnk8355/perm/paper2026/outputs_training/expA_METOPB"
)

output_path_metopC = (
    "/home/dnk8355/perm/paper2026/outputs_training/expA_METOPC"
)


pattern = "*_8epochs*.nc"


tbobs_B, tbsim_B, diff_B = load_experiment(
    output_path_metopB,
    pattern
)

tbobs_C, tbsim_C, diff_C = load_experiment(
    output_path_metopC,
    pattern
)

tbobs_B_day, diff_B_day = select_day_channel(
    tbobs_B,
    diff_B,
    target_date,
    channel
)

tbobs_C_day, diff_C_day = select_day_channel(
    tbobs_C,
    diff_C,
    target_date,
    channel
)

tbobs_B_NH, diff_B_NH = select_hemisphere(
    tbobs_B_day, diff_B_day, "NH"
)

tbobs_C_NH, diff_C_NH = select_hemisphere(
    tbobs_C_day, diff_C_day, "NH"
)


tbobs_B_SH, diff_B_SH = select_hemisphere(
    tbobs_B_day, diff_B_day, "SH"
)

tbobs_C_SH, diff_C_SH = select_hemisphere(
    tbobs_C_day, diff_C_day, "SH"
)

def get_max_abs(*diffs):
    """
    Get maximum absolute value ignoring unavailable datasets.
    """
    values = [
        float(np.abs(d).max())
        for d in diffs
        if d is not None
    ]

    return max(values)

max_abs_NH = get_max_abs(
    diff_B_NH,
    diff_C_NH
)

norm_NH = TwoSlopeNorm(
    vmin=-max_abs_NH,
    vcenter=0,
    vmax=max_abs_NH
)


max_abs_SH = get_max_abs(
    diff_B_SH,
    diff_C_SH
)

norm_SH = TwoSlopeNorm(
    vmin=-max_abs_SH,
    vcenter=0,
    vmax=max_abs_SH
)



# -----------------------------------------------------
# Plot settings
# -----------------------------------------------------
# Figure: METOP-B/C NH and SH comparison
# -----------------------------------------------------
fig = plt.figure(figsize=(14, 16))

axes = [
    fig.add_subplot(2,2,1, projection=ccrs.NorthPolarStereo()),
    fig.add_subplot(2,2,2, projection=ccrs.NorthPolarStereo()),
    fig.add_subplot(2,2,3, projection=ccrs.SouthPolarStereo()),
    fig.add_subplot(2,2,4, projection=ccrs.SouthPolarStereo()),
]

polar_set_latlim([44,90], axes[0])
polar_set_latlim([44,90], axes[1])
polar_set_latlim([-90,-44], axes[2])
polar_set_latlim([-90,-44], axes[3])


# Plot
plot_difference_map(
    axes[0], tbobs_B_NH, diff_B_NH, norm_NH
)

plot_difference_map(
    axes[1], tbobs_C_NH, diff_C_NH, norm_NH
)

plot_difference_map(
    axes[2], tbobs_B_SH, diff_B_SH, norm_SH
)

plot_difference_map(
    axes[3], tbobs_C_SH, diff_C_SH, norm_SH
)


axes[0].set_title("METOP-B NH", fontsize=14)
axes[1].set_title("METOP-C NH", fontsize=14)
axes[2].set_title("METOP-B SH", fontsize=14)
axes[3].set_title("METOP-C SH", fontsize=14)


# -----------------------------------------------------
# Common colourbars for each hemisphere
# -----------------------------------------------------

sm_NH = plt.cm.ScalarMappable(
    cmap="RdBu_r",
    norm=norm_NH
)

sm_SH = plt.cm.ScalarMappable(
    cmap="RdBu_r",
    norm=norm_SH
)

# Manually create colourbar axes.
# Format: [left, bottom, width, height] in figure coordinates.

cax_NH = fig.add_axes([0.12, 0.49, 0.76, 0.015])
cax_SH = fig.add_axes([0.12, 0.02, 0.76, 0.015])

# Draw colourbars
cbar_NH = fig.colorbar(
    sm_NH,
    cax=cax_NH,
    orientation="horizontal"
)
cbar_NH.ax.tick_params(labelsize=14) 
cbar_NH.set_label("TBobs - TBsim (K) NH", fontsize=16)

cbar_SH = fig.colorbar(
    sm_SH,
    cax=cax_SH,
    orientation="horizontal"
)
cbar_SH.ax.tick_params(labelsize=14) 
cbar_SH.set_label("TBobs - TBsim (K) SH", fontsize=16)


fig.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.90,
    bottom=0.08,
    wspace=0.15,
    hspace=0.22
)


fig.suptitle(
    f"Brightness temperature differences\nChannel {channel} {channel_name}  - {target_date}",
    fontsize=18
)

plt.savefig(
    output_path_metopB + '/plots/obs_minus_sim_' + str(target_date) +'_channel'+str(channel)+'_common_scales.png',
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# -----------------------------------------------------
# Figure: each map with its own colour scale
# -----------------------------------------------------

def create_norm(diff):
    """
    Create an individual symmetric colour scale centered at zero.
    """

    if diff is None:
        return None

    max_abs = float(np.abs(diff).max())

    return TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs
    )

def plot_difference_map_with_colorbar(ax, tbobs_day, diff, title):
    """
    Plot map with its own colourbar and own normalization.
    """

    # If this channel is not available for this sensor,
    # leave the subplot empty.
    if tbobs_day is None or diff is None:
        ax.text(
            0.5,
            0.5,
            "Channel not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14
        )

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.gridlines()

        ax.set_title(title, fontsize=14)

        return


    # Create individual colour normalization
    norm = create_norm(diff)

    sc = ax.scatter(
        tbobs_day.lon,
        tbobs_day.lat,
        c=diff,
        s=15,
        cmap="RdBu_r",
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.gridlines()

    ax.set_title(title, fontsize=14)

    # Individual colourbar
    cbar = plt.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.05
    )

    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("TBobs - TBsim (K)", fontsize=16)

# -----------------------------------------------------
# Create figure
# -----------------------------------------------------

fig = plt.figure(figsize=(14, 14))

axes = [
    fig.add_subplot(2, 2, 1, projection=ccrs.NorthPolarStereo()),
    fig.add_subplot(2, 2, 2, projection=ccrs.NorthPolarStereo()),
    fig.add_subplot(2, 2, 3, projection=ccrs.SouthPolarStereo()),
    fig.add_subplot(2, 2, 4, projection=ccrs.SouthPolarStereo()),
]


# Polar limits

polar_set_latlim([44, 90], axes[0])
polar_set_latlim([44, 90], axes[1])
polar_set_latlim([-90, -44], axes[2])
polar_set_latlim([-90, -44], axes[3])


# -----------------------------------------------------
# Plot each case independently
# -----------------------------------------------------

plot_difference_map_with_colorbar(
    axes[0],
    tbobs_B_NH,
    diff_B_NH,
    "METOP-B NH"
)

plot_difference_map_with_colorbar(
    axes[1],
    tbobs_C_NH,
    diff_C_NH,
    "METOP-C NH"
)

plot_difference_map_with_colorbar(
    axes[2],
    tbobs_B_SH,
    diff_B_SH,
    "METOP-B SH"
)

plot_difference_map_with_colorbar(
    axes[3],
    tbobs_C_SH,
    diff_C_SH,
    "METOP-C SH"
)


fig.suptitle(
    f"Brightness temperature differences\nChannel {channel} {channel_name} - {target_date}",
    fontsize=18,
    y=0.98
)

plt.tight_layout()
plt.savefig(
    output_path_metopB + '/plots/obs_minus_sim_' + str(target_date) +'_channel'+str(channel) + '_individual_scales.png',
    dpi=300,
    bbox_inches="tight"
)
plt.show()




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

# Open my initial conditions for surface temperature 
# ifs_sic = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/ifs_seaice_initials_METOP-B_1apr2024_31march2025_without_land_without_nans.nc")
# ifs_tsfc = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/ifs_tsfc_METOP-B_1apr2024_31march2025_dailyx_without_land.nc")

output_path = "/home/dnk8355/perm/paper2026/outputs_training/expA_METOPB_def"
tag_name='1epochs'
#scanpos - scan positions for each observation
scanpos=xr.open_dataset("/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/METOP-B_SCANPOS.nc")
#FG_dep for each observation
fg_dep=xr.open_dataset("/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/METOP-B_FG_DEP.nc")

# Collect all .nc files in the folder
files = sorted(glob.glob(os.path.join(output_path, "*_1epochs*.nc")))

datasets = [xr.open_dataset(f) for f in files]

# models contain losses
models=datasets[1]


#Plot loss terms
plt.plot(
    models.epoch,
    models.loss,
    label='Total loss',
    color='black',
    linestyle='--',
    linewidth=2.5,zorder=5
)
plt.plot(
    models.epoch,
    models.loss_channel_weighted,
    label='Observation loss',
    color='green',
    linewidth=2.5
)
plt.plot(
    models.epoch,
    models.emis_loss * 100,
    label=r'Emissivity loss ($J_{\mathrm{emis}}$) $\times 10^{2}$',
    color='orange',
    linewidth=2.5
)
plt.plot(
    models.epoch,
    models.bias_loss * 100000,
    label=r'Bias loss ($J_{\mathrm{bias}}$) $\times 10^{5}$',
    color='cyan',
    linewidth=2.5
)
plt.plot(
    models.epoch,
    models.seaice_loss * 10,
    label=r'Sea ice loss ($J_{\mathrm{seaice\_bounds}} + J_{\mathrm{false\_sic}}$) $\times 10$',
    color='blue',
    linewidth=2.5
)
plt.plot(
    models.epoch,
    models.tsfc_loss * 1e7,
    label=r'Temperature loss ($J_{\mathrm{seaice\_tsfc}}$) $\times 10^{7}$',
    color='magenta',
    linewidth=2.5
)
# Axis labels
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
# Ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.minorticks_on()
plt.tick_params(which='both', direction='in', length=6)
plt.tick_params(which='minor', length=3)
# Legend
plt.legend(fontsize=12, frameon=False)
plt.gca().margins(x=0, y=0)
# Limits
plt.ylim(0, 13)
plt.tight_layout()
plt.title('bg_emis=0.7 for 23.8GHz')
plt.savefig(f"{output_path}/plots/losses_bg"+tag_name+'.png', dpi=300, bbox_inches="tight")
plt.show()


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
results.to_csv(f"{output_path}/statistics_{tag_name}.csv")


#def polarCentral_set_latlim(lat_lims, ax):
#    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
#    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
#    theta = np.linspace(0, 2*np.pi, 100)
#    center, radius = [0.5, 0.5], 0.5
#    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#    circle = mpath.Path(verts * radius + center)


#Day of interest
target_date = np.datetime64('2023-04-01')
# Filter the dataset for the target date
tbobs_filtered = tbobs.where(tbobs['julian_day'].dt.floor('D') == target_date, drop=True)
tbsim_filtered = tbsim.where(tbsim['julian_day'].dt.floor('D') == target_date, drop=True)
tbsim_ini_filtered = tbsim_ini.where(tbsim_ini['julian_day'].dt.floor('D') == target_date, drop=True)

target_date_odb = np.datetime64('2024-04-01')
fg_dep_filtered = fg_dep.where(fg_dep.obs.dt.floor('D') == target_date_odb, drop=True)

lat_lims = [50, 90]

# Define channel names
channel_names = ['23V', '31V', '50V', '53V']

data_types = ['Initial guess', 'Analysis', 'Observations', 'Initial - obs', 'Analysis - obs']
datasets_list = [tbsim_ini_filtered, tbsim_filtered, tbobs_filtered]


# --- Create figure 4x5 ---
# Plot simulated and observed BT for the different channels
# and differences
fig, axes = plt.subplots(
    4, 5, figsize=(18, 18),
    subplot_kw={'projection': ccrs.NorthPolarStereo()}
)

# --- Layout ---
plt.subplots_adjust(
    left=0.08, right=0.96,
    top=0.92, bottom=0.06,
    wspace=0.04, hspace=0.18
)

for ch in range(4):  # rows -> channels

    # --- Common limits for absolute TBs ---
    vmin = min(
        tbsim_ini_filtered.isel(channel=ch).tb.min().item(),
        tbsim_filtered.isel(channel=ch).tb.min().item(),
        tbobs_filtered.isel(channel=ch).tb.min().item()
    )
    vmax = max(
        tbsim_ini_filtered.isel(channel=ch).tb.max().item(),
        tbsim_filtered.isel(channel=ch).tb.max().item(),
        tbobs_filtered.isel(channel=ch).tb.max().item()
    )

    # --- Differences ---
    diff_sim_ini = (
        tbsim_ini_filtered.isel(channel=ch).tb
        - tbobs_filtered.isel(channel=ch).tb
    )
    diff_sim = (
        tbsim_filtered.isel(channel=ch).tb
        - tbobs_filtered.isel(channel=ch).tb
    )

    zero_pos_sim_ini=(0-diff_sim_ini.min())/(diff_sim_ini.max()-diff_sim_ini.min())
    zero_pos_sim=(0-diff_sim.min())/(diff_sim.max()-diff_sim.min())
    cmap_custom_sim_ini = LinearSegmentedColormap.from_list(
        'blue_white_red',
        [(0.0, 'navy'), (zero_pos_sim_ini, 'white'), (1.0, 'darkred')]
    )
    cmap_custom_sim = LinearSegmentedColormap.from_list(
        'blue_white_red',
        [(0.0, 'navy'), (zero_pos_sim,'white'), (1.0, 'darkred')]
    )
    sc_abs = None
    sc_diff = None

    for col in range(5):
        ax = axes[ch, col]
        polarCentral_set_latlim(lat_lims, ax)

        # --- Absolute TBs ---
        if col < 3:
            tb_data = datasets_list[col].isel(channel=ch)
            sc = ax.scatter(
                tb_data.lon, tb_data.lat,
                c=tb_data.tb,
                s=15,
                cmap='jet',
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree()
            )
            if sc_abs is None:
                sc_abs = sc

        # --- sim_ini - obs ---
        elif col == 3:
            sc_diff_sim = ax.scatter(
                tbobs_filtered.lon,
                tbobs_filtered.lat,
                c=diff_sim_ini,
                s=15,
                cmap=cmap_custom_sim_ini,
                norm=Normalize(vmin=diff_sim_ini.min(),vmax=diff_sim_ini.max()),
                #vmin=diff_sim_ini.min(),
                #vmax=diff_sim_ini.max(),
                transform=ccrs.PlateCarree()
            )

        # --- sim - obs ---
        else:
            sc_diff_sim_ini = ax.scatter(
                tbobs_filtered.lon,
                tbobs_filtered.lat,
                c=diff_sim,
                s=15,
                cmap=cmap_custom_sim,
                norm=Normalize(vmin=diff_sim.min(),vmax=diff_sim.max()),
                #vmin=diff_sim.min(),
                #vmax=diff_sim.max(),
                transform=ccrs.PlateCarree()
            )

        # --- Map features ---
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines(alpha=0.5)

        if ch == 0:
            ax.set_title(data_types[col], fontsize=22, pad=12)

    # ========= COLORBARS =========

    pos0 = axes[ch, 0].get_position()
    pos2 = axes[ch, 2].get_position()
    pos3 = axes[ch, 3].get_position()
    pos4 = axes[ch, 4].get_position()

    cbar_height = 0.015
    cbar_y = pos0.y0 - 0.035

    # --- Absolute TB colorbar (cols 0–2) ---
    cax_abs = fig.add_axes([
        pos0.x0,
        cbar_y,
        pos2.x1 - pos0.x0,
        cbar_height
    ])
    cbar_abs = fig.colorbar(sc_abs, cax=cax_abs, orientation='horizontal')
    cbar_abs.ax.tick_params(labelsize=16)
    cbar_abs.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    cbar_abs.set_label('Brightness temp. (K)', fontsize=16, labelpad=-57)

   # --- sim_ini - obs colorbar (col 3) ---
    cax_diff_ini = fig.add_axes([
        pos3.x0,
        cbar_y,
        pos3.width,
        cbar_height
    ])
    cbar_diff_ini = fig.colorbar(sc_diff_sim, cax=cax_diff_ini, orientation='horizontal')
    cbar_diff_ini.ax.tick_params(labelsize=16)
    cbar_diff_ini.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    cbar_diff_ini.set_label('Brightness temp. (K)', fontsize=16, labelpad=-57)

    # --- sim - obs colorbar (col 4) ---
    cax_diff = fig.add_axes([
        pos4.x0,
        cbar_y,
        pos4.width,
        cbar_height
    ])
    cbar_diff = fig.colorbar(sc_diff_sim_ini, cax=cax_diff, orientation='horizontal')
    cbar_diff.ax.tick_params(labelsize=16)
    cbar_diff.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    cbar_diff.set_label('Brightness temp. (K)', fontsize=16, labelpad=-57)


# ---- Row labels ----
for ch, name in enumerate(channel_names):
    pos = axes[ch, 0].get_position()
    fig.text(
        pos.x0 - 0.015,
        pos.y0 + pos.height / 2,
        name,
        va='center', ha='center',
        fontsize=24, rotation=90
    )

# ---- Overall title ----
fig.suptitle('Brightness Temperatures '+'01/04/2024' + ', bg_emis=0.6 for 23V', fontsize=32)
fig.savefig(f"{output_path}/BT_01042024_bg_emis"+tag_name+'.png', dpi=300, bbox_inches="tight")

plt.show()



# --- Create figure 2x4 ---
# Plot simulated and observed BT for the different channels
fig, axes = sp_map(2, 4, figsize=(20, 10))

# --- Loop over each channel / column ---
for ch in range(4):
    ax_top = axes[0, ch]    # Observed
    ax_bottom = axes[1, ch] # Simulated

    # Select channel
    tb_obs_ch = tbobs_filtered.isel(channel=ch)
    tb_sim_ch = tbsim_filtered.isel(channel=ch)

    # Determine common color limits for this column
    vmin = min(tb_obs_ch.tb.min().item(), tb_sim_ch.tb.min().item())
    vmax = max(tb_obs_ch.tb.max().item(), tb_sim_ch.tb.max().item())

    # --- Top row: Observed ---
    polarCentral_set_latlim(lat_lims, ax_top)
    cs_top = ax_top.scatter(
        tb_obs_ch.lon, tb_obs_ch.lat, c=tb_obs_ch.tb,
        s=1, cmap='jet', transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax
    )
    cbar_top = plt.colorbar(cs_top, ax=ax_top, orientation='horizontal', shrink=0.9, pad=0.03)
    cbar_top.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    cbar_top.ax.tick_params(labelsize=25)
    ax_top.add_feature(cfeature.COASTLINE)
    ax_top.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
    ax_top.add_feature(cfeature.LAND, facecolor='gray')
    ax_top.gridlines()
    ax_top.set_title(f'Observed BT {channel_names[ch]}', fontsize=27)

    # --- Bottom row: Simulated ---
    polarCentral_set_latlim(lat_lims, ax_bottom)
    cs_bottom = ax_bottom.scatter(
        tb_sim_ch.lon, tb_sim_ch.lat, c=tb_sim_ch.tb,
        s=1, cmap='jet', transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax
    )
    cbar_bottom = plt.colorbar(cs_bottom, ax=ax_bottom, orientation='horizontal', shrink=0.9, pad=0.03)
    cbar_bottom.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    cbar_bottom.ax.tick_params(labelsize=25)
    ax_bottom.add_feature(cfeature.COASTLINE)
    ax_bottom.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
    ax_bottom.add_feature(cfeature.LAND, facecolor='gray')
    ax_bottom.gridlines()
    ax_bottom.set_title(f'Simulated BT {channel_names[ch]}', fontsize=27)

plt.tight_layout()
plt.show()



# Open my initial conditions for sea ice
ifs_sic_without_land = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/ifs_seaice_initials_METOP-B_1apr2024_31march2025_without_land_without_nans.nc")
plt.scatter(ifs_sic_without_land.LON, ifs_sic_without_land.LAT,c=ifs_sic_without_land.SEAICE[:,0])

# Open daily sea ice from IFS
daily_ifs_sic_without_land_without_nans = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/ifs_seaice_METOP-B_1apr2024_31march2025_daily_without_land_without_nans.nc")
daily_ifs_sic_without_land = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/ifs_seaice_METOP-B_1apr2024_31march2025_daily_without_land.nc")

plt.scatter(daily_ifs_sic_without_land_without_nans.LON, daily_ifs_sic_without_land_without_nans.LAT,c=daily_ifs_sic_without_land_without_nans.SIC[:,0])


def day_index(date_str, start_str="2024-04-01"):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start = datetime.strptime(start_str, "%Y-%m-%d")
    return (date - start).days 

day=day_index("2025-02-01")

#Plot sea ice concentration only 1 day
# --- First dataset: analysis ---
seaice_analysis = datasets[4].seaice.isel(lagstep=day).values
lon1 = datasets[4].lon.values
lat1 = datasets[4].lat.values

# --- Second dataset: IFS initial conditions ---
seaice_ifs = daily_ifs_sic_without_land_without_nans.SIC[:, day].values
lon2 = daily_ifs_sic_without_land_without_nans.LON.values
lat2 = daily_ifs_sic_without_land_without_nans.LAT.values

# --- Difference ---
seaice_diff = seaice_analysis - seaice_ifs

sic_min = min(seaice_analysis.min(), seaice_ifs.min())
sic_max = max(seaice_analysis.max(), seaice_ifs.max())

norm_ice = Normalize(vmin=sic_min, vmax=sic_max)
# --- Colormap: blue to white ---
colors = [(0.0, 0.0, 0.3), (0.0, 0.2, 0.6), (0.5, 0.7, 1.0), (1.0, 1.0, 1.0)]
custom_cmap_ice = LinearSegmentedColormap.from_list('ice_cmap', colors, N=100)


zero_pos_sim=(0-seaice_diff.min())/(seaice_diff.max()-seaice_diff.min())
cmap_custom_sim = LinearSegmentedColormap.from_list(
    'blue_white_red',
    [(0.0, 'navy'), (zero_pos_sim, 'white'), (1.0, 'darkred')]
)

# --- Create figure with 3 subplots ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8.5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
lat_lims = [50, 90]

#Plot Analysis
ax = axes[0]
ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())

cs1 = ax.scatter(
    lon1, lat1,
    c=seaice_analysis,
    s=70,
    cmap=custom_cmap_ice,
    norm=norm_ice,
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='gray')
ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
ax.gridlines()
ax.set_title('Sea Ice Concentration – Analysis \n 01/02/2025', fontsize=24)

ax = axes[1]
ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())

cs2 = ax.scatter(
    lon2, lat2,
    c=seaice_ifs,
    s=70,
    cmap=custom_cmap_ice,
    norm=norm_ice,
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='gray')
ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
ax.gridlines()
ax.set_title('Sea Ice Concentration – IFS \n 01/02/2025', fontsize=24)

ax = axes[2]
ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())

cs3 = ax.scatter(
    lon1, lat1,
    c=seaice_diff,
    s=60,
    cmap=cmap_custom_sim,
    norm=Normalize(vmin=seaice_diff.min(),vmax=seaice_diff.max()),
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='gray')
ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
ax.gridlines()
ax.set_title('Sea Ice Difference (Analysis − IFS) \n 01/02/2025', fontsize=24)

# === axis ===
pos0 = axes[0].get_position()
pos1 = axes[1].get_position()
pos2 = axes[2].get_position()

cbar_height = 0.025
cbar_y = pos0.y0 - 0.10   

# --- Shared colorbar for first two plots ---
cax1 = fig.add_axes([
    pos0.x0,             
    cbar_y,
    pos1.x1 - pos0.x0,   
    cbar_height
])

cbar1 = fig.colorbar(cs1, cax=cax1, orientation='horizontal')
cbar1.set_label('Sea Ice Concentration', fontsize=22, labelpad=-65)
cbar1.ax.tick_params(labelsize=20)

# --- Colorbar for difference plot ---
cax2 = fig.add_axes([
    pos2.x0,
    cbar_y,
    pos2.width,
    cbar_height
])

cbar2 = fig.colorbar(cs3, cax=cax2, orientation='horizontal')
cbar2.set_label('Sea Ice Concentration Difference', fontsize=22, labelpad=-65)
cbar2.ax.tick_params(labelsize=20)
fig.savefig(f"{output_path}/SIC_01022025_bg_emis"+tag_name+'.png', dpi=300, bbox_inches="tight")


#fig.suptitle(
#    '1 April 2024',
#    fontsize=28,
#    y=0.95
#)


#Plot SIC 4 days with differences with first day (NOT FINISHED)
nrows = 4
ncols = 3
days = [day + i for i in range(nrows)]

fig = plt.figure(figsize=(24, 4 * nrows))
gs = gridspec.GridSpec(nrows, ncols, hspace=0.35, wspace=0.05)
lon1 = datasets[4].lon.values
lat1 = datasets[4].lat.values
lon2 = daily_ifs_sic_without_land_without_nans.LON.values
lat2 = daily_ifs_sic_without_land_without_nans.LAT.values
# --- Colormap: blue to white ---
colors = [(0.0, 0.0, 0.3), (0.0, 0.2, 0.6), (0.5, 0.7, 1.0), (1.0, 1.0, 1.0)]
custom_cmap_ice = LinearSegmentedColormap.from_list('ice_cmap', colors, N=100)

for i, d in enumerate(days):

    # --- Data for this day ---
    seaice_analysis = datasets[4].seaice.isel(lagstep=d).values
    seaice_ifs = daily_ifs_sic_without_land_without_nans.SIC[:, d].values
    seaice_diff = seaice_analysis - seaice_ifs
    zero_pos_sim=(0-seaice_diff.min())/(seaice_diff.max()-seaice_diff.min())
    cmap_custom_sim = LinearSegmentedColormap.from_list(
        'blue_white_red',
        [(0.0, 'navy'), (zero_pos_sim, 'white'), (1.0, 'darkred')]
    )
    # --- Date label ---
    date = datetime(2025, 2, 1) + timedelta(days=d)
    date_str = date.strftime("%d/%m/%Y")

    # ---------- Column 1: Analysis ----------
    ax0 = fig.add_subplot(gs[i, 0], projection=ccrs.NorthPolarStereo())
    ax0.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

    cs1 = ax0.scatter(
        lon1, lat1,
        c=seaice_analysis,
        s=30,
        cmap=custom_cmap_ice,
        norm= Normalize(vmin=min(seaice_analysis.min(), seaice_ifs.min()), vmax=max(seaice_analysis.max(), seaice_ifs.max())),
        transform=ccrs.PlateCarree()
    )

    ax0.add_feature(cfeature.COASTLINE)
    ax0.add_feature(cfeature.LAND, facecolor='gray')
    ax0.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
    ax0.gridlines()

    if i == 0:
        ax0.set_title("Sea Ice Concentration – Analysis", fontsize=22)
    ax0.text(0.5, 1.02, date_str, transform=ax.transAxes,
            ha='center', fontsize=18)

    # ---------- Column 2: IFS ----------
    ax1 = fig.add_subplot(gs[i, 1], projection=ccrs.NorthPolarStereo())
    ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

    ax1.scatter(
        lon2, lat2,
        c=seaice_ifs,
        s=30,
        cmap=custom_cmap_ice,
        norm=Normalize(vmin=min(seaice_analysis.min(), seaice_ifs.min()), vmax=max(seaice_analysis.max(), seaice_ifs.max())),
        transform=ccrs.PlateCarree()
    )

    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.LAND, facecolor='gray')
    ax1.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
    ax1.gridlines()

    if i == 0:
        ax1.set_title("Sea Ice Concentration – IFS", fontsize=22)

    # ---------- Column 3: Difference ----------
    ax2 = fig.add_subplot(gs[i, 2], projection=ccrs.NorthPolarStereo())
    ax2.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

    cs3 = ax2.scatter(
        lon1, lat1,
        c=seaice_diff,
        s=30,
        cmap=cmap_custom_sim,
        norm=Normalize(vmin=seaice_diff.min(), vmax=seaice_diff.max()),
        transform=ccrs.PlateCarree()
    )

    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.LAND, facecolor='gray')
    ax2.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
    ax2.gridlines()

    if i == 0:
        ax2.set_title("Sea Ice Difference (Analysis − IFS)", fontsize=22)

    # ==========================================================
    # >>> COLORBARS FOR THIS ROW <<<
    # ==========================================================
    pos0 = ax0.get_position()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    cbar_height = 0.015
    cbar_y = pos0.y0 - 0.03

    # --- Shared colorbar (Analysis + IFS) ---
    cax1 = fig.add_axes([
        pos0.x0,
        cbar_y,
        pos1.x1 - pos0.x0,
        cbar_height
    ])

    cbar1 = fig.colorbar(cs1, cax=cax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=12)

    # --- Difference colorbar ---
    cax2 = fig.add_axes([
        pos2.x0,
        cbar_y,
        pos2.width,
        cbar_height
    ])

    cbar2 = fig.colorbar(cs3, cax=cax2, orientation='horizontal')
    cbar2.ax.tick_params(labelsize=12)



#Plot of SIC of 8 consecutive days from datasets[4]
# --- Prepare colormap and normalization ---
colors = [(0.0, 0.0, 0.3), (0.0, 0.2, 0.6), (0.5, 0.7, 1.0), (1.0, 1.0, 1.0)]
custom_cmap_ice = LinearSegmentedColormap.from_list('ice_cmap', colors, N=100)
norm_ice = BoundaryNorm(np.linspace(0, 1, 101), ncolors=100)

# --- Extract lat/lon ---
lon = datasets[4].lon.values
lat = datasets[4].lat.values

# --- Select first 8 days ---
n_days = 8
seaice_8days = datasets[4].seaice.isel(lagstep=slice(0, n_days))

# --- Figure setup: 2 rows x 4 columns ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': ccrs.NorthPolarStereo()})
lat_lims = [50, 90]

for i in range(n_days):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    cs = ax.scatter(lon, lat, c=seaice_8days.isel(lagstep=i), s=20,
                    cmap=custom_cmap_ice, norm=norm_ice, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='gray')
    ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
    ax.gridlines()
    ax.set_title(f'Day {i+1}', fontsize=12)

# --- Shared horizontal colorbar below all subplots ---
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # left, bottom, width, height
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Sea Ice Concentration', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()




#################################################
#Plot properties 1 and 2 over 3 consecutive days
###################################################
# --- Custom color map ---
colors = [
    (0.0, 0.0, 0.6), (0.0, 0.2, 0.8), (0.0, 0.4, 1.0), (0.2, 0.6, 1.0),
    (0.0, 0.6, 0.4), (0.0, 0.8, 0.3), (0.2, 0.9, 0.2), (0.6, 1.0, 0.2),
    (1.0, 1.0, 0.4), (1.0, 0.8, 0.2), (1.0, 0.6, 0.0), (1.0, 0.4, 0.0),
    (1.0, 0.2, 0.0), (0.9, 0.0, 0.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0)
]
custom_cmap = ListedColormap(colors)

def day_index(date_str, start_str="2024-04-01"):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start = datetime.strptime(start_str, "%Y-%m-%d")
    return (date - start).days 

day=day_index("2025-02-01")
days = [day + i for i in range(4)]
days_nb=["01/02/2025","02/02/2025","03/02/2025"]

# --- Extract lat/lon ---
lon = datasets[2].lon.values
lat = datasets[2].lat.values


# --- Compute min/max per property over first 3 steps ---
prop1_min = datasets[2].properties.isel(prop=0, step=slice(days[0],days[-1])).min().item()
prop1_max = datasets[2].properties.isel(prop=0, step=slice(days[0],days[-1])).max().item()

prop2_min = datasets[2].properties.isel(prop=1, step=slice(days[0],days[-1])).min().item()
prop2_max = datasets[2].properties.isel(prop=1, step=slice(days[0],days[-1])).max().item()

norm1 = BoundaryNorm(np.linspace(prop1_min, prop1_max, len(colors)), ncolors=len(colors))
norm2 = BoundaryNorm(np.linspace(prop2_min, prop2_max, len(colors)), ncolors=len(colors))

# --- Figure setup: 3 rows x 2 columns ---
fig, axes = plt.subplots(3, 2, figsize=(16, 20), subplot_kw={'projection': ccrs.NorthPolarStereo()})
lat_lims = [50, 90]

# --- Loop over steps and properties ---
for row in range(3):
    for col in range(2):
        ax = axes[row, col]
        ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
        
        # Select the corresponding data: properties(grid, step, prop)
        data = datasets[2].properties.isel(step=row, prop=col)
        
        # Choose correct norm for column
        norm = norm1 if col == 0 else norm2
        
        # Scatter plot
        sc = ax.scatter(lon, lat, c=data, s=20, cmap=custom_cmap, norm=norm, transform=ccrs.PlateCarree())
        
        # Features and grid
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.gridlines()
        ax.set_title('Property '+str(col+1) + '-' + days_nb[row] , fontsize=24)

# --- Add shared colorbars for each column dynamically ---
# Get positions of the bottom row axes for each column
pos_col1 = axes[-1,0].get_position()  # last row, first column
pos_col2 = axes[-1,1].get_position()  # last row, second column

cbar_height = 0.03
cbar_y = pos_col1.y0 - 0.05  # just below the bottom row

# Colorbar for column 1
cbar_ax1 = fig.add_axes([
    pos_col1.x0,
    cbar_y,
    pos_col1.width,
    cbar_height
])
cbar1=fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm1), 
             cax=cbar_ax1, orientation='horizontal')
cbar1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))  
cbar1.ax.tick_params(labelsize=16) 

# Colorbar for column 2
cbar_ax2 = fig.add_axes([
    pos_col2.x0,
    cbar_y,
    pos_col2.width,
    cbar_height
])
cbar2=fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm2), 
             cax=cbar_ax2, orientation='horizontal')
cbar2.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))  
cbar2.ax.tick_params(labelsize=16) 
fig.savefig(f"{output_path}/seaice_proper_01022025_03022025_bg_emis"+tag_name+'.png', dpi=300, bbox_inches="tight")




# Open reduced grid files created after removing land points
reduced_grid_ds = xr.open_dataset("/perm/dnk8355/odb_files_test/METOP-B_1april2024_31march2025_lat_lon_corrected_ref_above50_without_land.nc")
lons_ref = reduced_grid_ds.lon.values
lats_ref = reduced_grid_ds.lat.values

plt.scatter(lons_ref,lats_ref,c=datasets[1].properties[:,1,1].values)
plt.scatter(lons_ref,lats_ref,c=datasets[1].properties[:,3,1].values)
#file="/perm/dnk8355/sea_ice_data/tbsim_year.nc"
#ds = xr.open_dataset(file)

plt.scatter(datasets[0].epoch,datasets[0].loss)

reduced_lats = xr.open_dataset("/perm/dnk8355/netcdf_monthly_feb2025/METOP-B_202502_LAT.nc")
reduced_lons = xr.open_dataset("/perm/dnk8355/netcdf_monthly_feb2025/METOP-B_202502_LON.nc")

plt.scatter(reduced_lons.LON,reduced_lats.LAT,c=datasets[1].properties[:,1,1].values)
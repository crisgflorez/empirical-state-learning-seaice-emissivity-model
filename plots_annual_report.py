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


#Plot utilities
def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5 #0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)


def sp_map(*nrs, projection=ccrs.NorthPolarStereo(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection': projection}, **kwargs)

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

#Compute statistics table
def compute_tb_statistics(folder, tag, scanpos, fg_dep):

    exp = load_experiment(folder, tag)
    tbobs = exp["tbobs"]
    tbsim = exp["tbsim"]

    # Scanpos masks
    mask_center = scanpos["SCANPOS"].isin([14, 15, 16])
    mask_edge = (
        ((scanpos["SCANPOS"] >= 0) & (scanpos["SCANPOS"] <= 5)) |
        ((scanpos["SCANPOS"] >= 25) & (scanpos["SCANPOS"] <= 30))
    )

    fg_dep_c = fg_dep.where(mask_center, drop=True)
    fg_dep_e = fg_dep.where(mask_edge, drop=True)

    mask_center = mask_center.rename({"obs": "iobs"})
    mask_edge = mask_edge.rename({"obs": "iobs"})

    tbobs_c = tbobs.where(mask_center, drop=True)
    tbsim_c = tbsim.where(mask_center, drop=True)

    tbobs_e = tbobs.where(mask_edge, drop=True)
    tbsim_e = tbsim.where(mask_edge, drop=True)

    # Differences
    diff_all = tbobs["tb"] - tbsim["tb"]
    diff_c = tbobs_c["tb"] - tbsim_c["tb"]
    diff_e = tbobs_e["tb"] - tbsim_e["tb"]

    # Stats
    stats = {
        "Mean diff (all)": diff_all.mean("iobs"),
        "Mean diff IFS (all)": fg_dep['FG_DEP'].mean(dim="obs"),
        "Mean diff (scanpos 14–16)": diff_c.mean("iobs"),
        "Mean diff IFS (scanpos 14–16)": fg_dep_c['FG_DEP'].mean("obs"),
        "Mean diff (edge)": diff_e.mean("iobs"),
        "Mean diff IFS (edge)": fg_dep_e['FG_DEP'].mean("obs"), 

        "STD diff (all)": diff_all.std("iobs"),
        "STD diff (scanpos 14–16)": diff_c.std("iobs"),
        "STD diff (edge)": diff_e.std("iobs"),

        "RMSE (all)": np.sqrt((diff_all**2).mean("iobs")),
        "RMSE (scanpos 14–16)": np.sqrt((diff_c**2).mean("iobs")),
        "RMSE (edge)": np.sqrt((diff_e**2).mean("iobs")),
    }

    channel_dim = list(stats.values())[0].dims[0]
    channels = stats["Mean diff (all)"][channel_dim].values

    table = pd.DataFrame(
        [v.values for v in stats.values()],
        index=list(stats.keys()),
        columns=channels
    )

    return table

def format_tb_statistics_table(table):
    """
    Create a table with:
    - shared labels via LaTeX multirow
    - scan position column
    - IFS values in brackets for mean rows
    """

    # Explicit channel names
    channels = ["23V", "31V", "50V", "53V"]

    rows = []

    def mean_block():
        rows.extend([
            (
                r"\multirow{3}{*}{(Obs$-$Sim).mean()}",
                "All",
                [f"{table.loc['Mean diff (all)', i]:.2f} "
                 f"[{table.loc['Mean diff IFS (all)', i]:.2f}\\textsuperscript{{†}}]" for i in range(4)]
            ),
            (
                "",
                "Around nadir",
                [f"{table.loc['Mean diff (scanpos 14–16)', i]:.2f} "
                 f"[{table.loc['Mean diff IFS (scanpos 14–16)', i]:.2f}]" for i in range(4)]
            ),
            (
                "",
                "Large angles",
                [f"{table.loc['Mean diff (edge)', i]:.2f} "
                 f"[{table.loc['Mean diff IFS (edge)', i]:.2f}]" for i in range(4)]
            ),
        ])

    def std_block():
        rows.extend([
            (
                r"\multirow{3}{*}{(Obs$-$Sim).std()}",
                "All",
                [f"{table.loc['STD diff (all)', i]:.2f}" for i in range(4)]
            ),
            (
                "",
                "Around nadir",
                [f"{table.loc['STD diff (scanpos 14–16)', i]:.2f}" for i in range(4)]
            ),
            (
                "",
                "Large angles",
                [f"{table.loc['STD diff (edge)', i]:.2f}" for i in range(4)]
            ),
        ])

    def rmse_block():
        rows.extend([
            (
                r"\multirow{3}{*}{RMSE}",
                "All",
                [f"{table.loc['RMSE (all)', i]:.2f}" for i in range(4)]
            ),
            (
                "",
                "Around nadir",
                [f"{table.loc['RMSE (scanpos 14–16)', i]:.2f}" for i in range(4)]
            ),
            (
                "",
                "Large angles",
                [f"{table.loc['RMSE (edge)', i]:.2f}" for i in range(4)]
            ),
        ])

    mean_block()
    std_block()
    rmse_block()

    df = pd.DataFrame(
        [[stat, scan] + vals for stat, scan, vals in rows],
        columns=["Statistic", "Scan positions"] + channels
    )

    return df

def save_tb_statistics_latex(table_fmt, outdir, tag, name):

    tex = table_fmt.to_latex(
        index=False,
        escape=False,
        column_format="llcccc"
    )

    lines = tex.splitlines()
    new_lines = []

    data_rows = 0
    in_table = False

    for line in lines:
        new_lines.append(line)

        if line.startswith("\\toprule"):
            in_table = True
            continue

        if in_table and "&" in line and not line.startswith("\\"):
            data_rows += 1
            if data_rows in (3, 6):
                new_lines.append("\\hline")

    tex = "\n".join(new_lines)

    outfile = os.path.join(outdir, f"table_tb_stats_{tag}.tex")
    with open(outfile, "w") as f:
        f.write(tex)

    print(f"Saved LaTeX table to {outfile}")

# Plot maps for one experiment
def plot_tb_maps(tbobs_f, tbsim_f, fg_dep_f, title, outfile):

    channel_names = ["23V", "31V", "50V", "53V"]
    lat_lims = [50, 90]

    # Plot simulated and observed BT for the different channels
    fig, axes = sp_map(4, 4, figsize=(9, 10))
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.96,
        wspace=0,
        hspace=0.05
    )
    row_labels = [
    "Observed BT [K]",
    "Simulated BT [K]",
    "Obs − Sim [K]",
    "FG departure [K]"]

    # --- Loop over each channel / column ---
    for ch in range(4):

        # Select channel
        tb_obs_ch = tbobs_f.isel(channel=ch)
        tb_sim_ch = tbsim_f.isel(channel=ch)
        tb_diff = tb_obs_ch.tb - tb_sim_ch.tb 
        fg_dep_ch = fg_dep_f.isel(channel=ch)

        # Determine common color limits for tbobs and tbsim
        vmin = min(tb_obs_ch.tb.min().item(), tb_sim_ch.tb.min().item())
        vmax = max(tb_obs_ch.tb.max().item(), tb_sim_ch.tb.max().item())


        # Determine common color limits for tbdiff and fg_dep_ch
        vmin_dif = min(tb_diff.min().item(), fg_dep_ch.FG_DEP.min().item())
        vmax_dif = max(tb_diff.max().item(), fg_dep_ch.FG_DEP.max().item())

        zero_pos_diff=(0-vmin_dif)/(vmax_dif-vmin_dif)
        cmap_custom_diff = LinearSegmentedColormap.from_list(
            'blue_white_red',
            [(0.0, 'navy'), (zero_pos_diff,'white'), (1.0, 'darkred')]
        )

        # ================= ROW 1: OBS =================
        ax = axes[0, ch]
        polarCentral_set_latlim(lat_lims, ax)
        im = ax.scatter(
            tb_obs_ch.lon, tb_obs_ch.lat,
            c=tb_obs_ch.tb, s=2,
            cmap="jet",
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.8, pad=0.03) #shrink=0.9
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar.ax.tick_params(labelsize=15)
        ax.set_title(channel_names[ch], fontsize=18,y=1.05)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines()

        # ================= ROW 2: SIM =================
        ax = axes[1, ch]
        polarCentral_set_latlim(lat_lims, ax)
        im = ax.scatter(
            tb_sim_ch.lon, tb_sim_ch.lat,
            c=tb_sim_ch.tb, s=2,
            cmap="jet",
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.8, pad=0.03) #shrink=0.9
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar.ax.tick_params(labelsize=15)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines()

        # ================= ROW 3: DIFF =================
        ax = axes[2, ch]
        polarCentral_set_latlim(lat_lims, ax)
        im = ax.scatter(
            tb_obs_ch.lon, tb_obs_ch.lat,
            c=tb_diff, s=2, cmap=cmap_custom_diff,
                norm=Normalize(vmin=vmin_dif,vmax=vmax_dif),
            transform=ccrs.PlateCarree()
        )
        cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.8, pad=0.03) #shrink=0.9
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar.set_ticks([vmin_dif, 0.0, vmax_dif])
        cbar.ax.tick_params(labelsize=15)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines()
        
        # ================= ROW 4: FG DEP =================
        ax = axes[3, ch]
        polarCentral_set_latlim(lat_lims, ax)
        im = ax.scatter(
            tb_obs_ch.lon, tb_obs_ch.lat,
            c=fg_dep_ch.FG_DEP, s=2,
            cmap=cmap_custom_diff,norm=Normalize(vmin=vmin_dif,vmax=vmax_dif),
            transform=ccrs.PlateCarree()
        )
        cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.8, pad=0.03) #shrink=0.9
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar.set_ticks([vmin_dif, 0.0, vmax_dif])
        cbar.ax.tick_params(labelsize=15)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines()

    # ----- ROW LABELS -----
    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            -0.25, 0.5, label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            fontsize=18
        )

    fig.suptitle(title, fontsize=22, y=1.05)
    letters = string.ascii_lowercase  # a, b, c, ...
    k = 0
    for i in range(axes.shape[0]):       # rows
        for j in range(axes.shape[1]):   # columns
            ax = axes[i, j]
            ax.text(
                0.01, 1,              # position inside axes
                f"{letters[k]})",
                transform=ax.transAxes,
                fontsize=14,
                va="top",
                ha="left"
            )
            k += 1
    fig.savefig(outfile, dpi=300, bbox_inches="tight")


# Plot maps for one experiment
def plot_tb_compare_experiments(tbobs_f, tbsim_all, title, outfile):

    channel_names = ["23V", "31V", "50V", "53V"]
    lat_lims = [50, 90]
    exp_names = list(tbsim_all.keys())   # ['exp1','exp2',...]
    n_exp = len(exp_names)

    # Plot simulated and observed BT for the different channels
    fig, axes = sp_map(n_exp + 1, 4, figsize=(7, 12))
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.96,
        wspace=0,
        hspace=0.1
    )
    row_labels = ["Observed BT\n       [K]"] + [f" Sim. BT\n{e} [K]" for e in exp_names]

    # --- Loop over each channel / column ---
    for ch in range(4):

        # Select channel
        tb_obs_ch = tbobs_f.isel(channel=ch)
       # --- collect simulated BT across all experiments ---
        tb_sim_ch_all = [
            tbsim_all[e].isel(channel=ch).tb
            for e in exp_names
        ]

        # --- common color limits (obs + all experiments) ---
        vmin = min(
            tb_obs_ch.tb.min().item(),
            min(tb.min().item() for tb in tb_sim_ch_all)
        )
        vmax = max(
            tb_obs_ch.tb.max().item(),
            max(tb.max().item() for tb in tb_sim_ch_all)
        )

        # ================= ROW 1: OBS =================
        ax = axes[0, ch]
        polarCentral_set_latlim(lat_lims, ax)
        im = ax.scatter(
            tb_obs_ch.lon, tb_obs_ch.lat,
            c=tb_obs_ch.tb, s=2,
            cmap="jet",
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.75, pad=0.03) #shrink=0.9
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar.ax.tick_params(labelsize=15)
        ax.set_title(channel_names[ch], fontsize=18,y=1.05)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines()


        # ================= ROWS 2–6: SIMULATED =================
        for i, exp in enumerate(exp_names):

            tb_sim_ch = tbsim_all[exp].isel(channel=ch)

            ax = axes[i + 1, ch]
            polarCentral_set_latlim(lat_lims, ax)

            im = ax.scatter(
                tb_sim_ch.lon,
                tb_sim_ch.lat,
                c=tb_sim_ch.tb,
                s=2,
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree()
            )
            cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.75, pad=0.03) #shrink=0.9
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            cbar.ax.tick_params(labelsize=15)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
            ax.add_feature(cfeature.LAND, facecolor='gray')
            ax.gridlines()

 
    # ----- ROW LABELS -----
    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            -0.45, 0.5, label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            fontsize=18
        )

    fig.suptitle(title, fontsize=22, y=1.03)
    letters = string.ascii_lowercase  # a, b, c, ...
    k = 0
    for i in range(axes.shape[0]):       # rows
        for j in range(axes.shape[1]):   # columns
            ax = axes[i, j]
            ax.text(
                0.01, 1,              # position inside axes
                f"{letters[k]})",
                transform=ax.transAxes,
                fontsize=14,
                va="top",
                ha="left"
            )
            k += 1
    fig.savefig(outfile, dpi=300, bbox_inches="tight")

def plot_tb_diff_compare_experiments(tbobs_f, tbsim_all, fg_dep_f, title, outfile):

    channel_names = ["23V", "31V", "50V", "53V"]
    lat_lims = [50, 90]
    exp_names = list(tbsim_all.keys())   # ['exp1','exp2',...]
    n_exp = len(exp_names)

    # Plot simulated and observed BT for the different channels
    fig, axes = sp_map(n_exp + 1, 4, figsize=(7, 12))
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.96,
        wspace=0,
        hspace=0.1
    )

    row_labels = ["FG departure\n       [ĸ]"] + [f"Obs − Sim \n {e} [K]" for e in exp_names]

    # --- Loop over each channel / column ---
    for ch in range(4):

        # Select channel
        tb_obs_ch = tbobs_f.isel(channel=ch)
        fg_dep_ch = fg_dep_f.isel(channel=ch)

        # --- compute Obs - Sim for all experiments ---
        tb_diff_all = [
            tb_obs_ch.tb - tbsim_all[e].isel(channel=ch).tb
            for e in exp_names
        ]

        # --- common limits (FG dep + all Obs-Sim) ---
        vmin = min(
            fg_dep_ch.FG_DEP.min().item(),
            min(diff.min().item() for diff in tb_diff_all)
        )
        vmax = max(
            fg_dep_ch.FG_DEP.max().item(),
            max(diff.max().item() for diff in tb_diff_all)
        )

        zero_pos_diff=(0-vmin)/(vmax-vmin)
        cmap_custom_diff = LinearSegmentedColormap.from_list(
            'blue_white_red',
            [(0.0, 'navy'), (zero_pos_diff,'white'), (1.0, 'darkred')]
        )

        # ================= ROW 1: FG dep =================
        ax = axes[0, ch]
        polarCentral_set_latlim(lat_lims, ax)
        im = ax.scatter(
            tb_obs_ch.lon, tb_obs_ch.lat,
            c=fg_dep_ch.FG_DEP, s=2,
            cmap=cmap_custom_diff,
            norm=Normalize(vmin=vmin,vmax=vmax),
            transform=ccrs.PlateCarree()
        )
        cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.75, pad=0.03) #shrink=0.9
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar.set_ticks([vmin, 0.0, vmax])
        cbar.ax.tick_params(labelsize=15)
        ax.set_title(channel_names[ch], fontsize=18,y=1.05)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.gridlines()


        # ================= ROWS 2–6: OBS - SIM =================
        for i, exp in enumerate(exp_names):

            ax = axes[i + 1, ch]
            polarCentral_set_latlim(lat_lims, ax)

            im = ax.scatter(
                tb_obs_ch.lon,
                tb_obs_ch.lat,
                c=tb_diff_all[i],
                s=2,
                cmap=cmap_custom_diff,
                norm=Normalize(vmin=vmin,vmax=vmax),
                transform=ccrs.PlateCarree()
            )
            cbar=plt.colorbar(im, ax=ax, orientation="horizontal",shrink=0.75, pad=0.03) #shrink=0.9
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            cbar.set_ticks([vmin, 0.0, vmax])
            cbar.ax.tick_params(labelsize=15)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
            ax.add_feature(cfeature.LAND, facecolor='gray')
            ax.gridlines()

 
    # ----- ROW LABELS -----
    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            -0.45, 0.5, label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            fontsize=18
        )

    fig.suptitle(title, fontsize=22, y=1.03)
    letters = string.ascii_lowercase  # a, b, c, ...
    k = 0
    for i in range(axes.shape[0]):       # rows
        for j in range(axes.shape[1]):   # columns
            ax = axes[i, j]
            ax.text(
                0.01, 1,              # position inside axes
                f"{letters[k]})",
                transform=ax.transAxes,
                fontsize=14,
                va="top",
                ha="left"
            )
            k += 1
    fig.savefig(outfile, dpi=300, bbox_inches="tight")

def plot_losses(models_all, exp_order, output_path):

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(exp_order),
        figsize=(16, 5),
        sharey=True
    )

    panel_labels = ['a)', 'b)', 'c)']  # labels for each subplot
    title_mapping = {
        'exp1': 'Experiment 1',
        'exp4': 'Experiment 4',
        'exp5': 'Experiment 5'
    }
    # Define bias_loss scaling per experiment
    bias_scaling = {
        'exp1': 10,
        'exp4': 1e5,
        'exp5': 1e5
    }

    # For Option 2, the legend will mention all scalings explicitly
    bias_legend_text = 'Bias loss ($J_{\\mathrm{bias}}$) × [10 for exp1, 1e5 for exp4&5]'

    for ax, exp_name, label in zip(axes, exp_order, panel_labels):
        models = models_all[exp_name]

        ax.plot(
            models.epoch,
            models.loss,
            label=r'Total loss ($J$)',
            color='black',
            linestyle='--',
            linewidth=2.5,
            zorder=5
        )
        ax.plot(
            models.epoch,
            models.loss_channel_weighted,
            label=r'Observation loss ($J_{\mathrm{obs}}$)',
            color='green',
            linewidth=2.5
        )
        ax.plot(
            models.epoch,
            models.emis_loss * 100,
            label=r'Emissivity loss ($J_{\mathrm{emis}}$) × 10²',
            color='orange',
            linewidth=2.5
        )
        # Bias loss with experiment-specific scaling
        scaling_factor = bias_scaling[exp_name]
        ax.plot(
            models.epoch,
            models.bias_loss * scaling_factor,
            label=bias_legend_text,
            color='cyan',
            linewidth=2.5
        )
        ax.plot(
            models.epoch,
            models.seaice_loss * 10,
            label=r'Sea ice loss ($J_{\mathrm{seaice\_bounds}} + J_{\mathrm{false\_sic}}$) × 10',
            color='blue',
            linewidth=2.5
        )
        ax.plot(
            models.epoch,
            models.tsfc_loss * 1e7,
            label=r'Temperature loss ($J_{\mathrm{seaice\_tsfc}}$) × 10⁷',
            color='magenta',
            linewidth=2.5
        )

        # Axis formatting
        ax.set_xlabel('Epoch', fontsize=18)
        ax.set_title(title_mapping[exp_name], fontsize=22)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(which='both', direction='in', length=6, top=True, right=True)
        ax.tick_params(which='minor', length=3, right=True)
        ax.tick_params(labelsize=15)
        ax.margins(x=0, y=0)
        ax.set_ylim(0, 120)

        # Panel label (a), b), c))
        ax.text(
            0.01, 1.1,
            label,
            transform=ax.transAxes,
            fontsize=18,
            va='top',
            ha='left'
        )

    axes[0].set_ylabel('Loss', fontsize=18)

    # Legend BELOW the plots (single legend)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=16,
        frameon=False,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.52, -0.15)
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(
        f"{output_path}/losses_exp1_exp4_exp5.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()



def filter_by_scanpos(tbsim_all, tbobs_f, scanpos_f):
    """
    Filter observed and simulated TB by scan position (center and edge).

    Parameters
    ----------
    tbsim_all : dict of xarray.Dataset
        Dictionary of simulated TB datasets for each experiment.
    tbobs_f : xarray.Dataset
        Observed TB dataset for the day.
    scanpos_f : xarray.Dataset
        Scan position dataset for the day (dimension 'obs').

    Returns
    -------
    tbobs_center, tbobs_edge : xarray.Dataset
        Observed TB filtered by center (14–16) and edge (0–5,25–30) scan positions.
    tbsim_center, tbsim_edge : dict of xarray.Dataset
        Simulated TB filtered by center and edge scan positions for each experiment.
    """

    # --- Create masks ---
    mask_center = scanpos_f["SCANPOS"].isin([14, 15, 16])
    mask_edge = ((scanpos_f["SCANPOS"] >= 0) & (scanpos_f["SCANPOS"] <= 5)) | \
                ((scanpos_f["SCANPOS"] >= 25) & (scanpos_f["SCANPOS"] <= 30))

    # --- Convert masks to have the same dimension as tbobs_f / tbsim_all ---
    mask_center_iobs = xr.DataArray(mask_center.values, dims="iobs")
    mask_edge_iobs = xr.DataArray(mask_edge.values, dims="iobs")

    # --- Filter tbobs ---
    tbobs_center = tbobs_f.where(mask_center_iobs, drop=True)
    tbobs_edge = tbobs_f.where(mask_edge_iobs, drop=True)

    # --- Filter tbsim_all ---
    tbsim_center = {}
    tbsim_edge = {}
    for key, ds in tbsim_all.items():
        tbsim_center[key] = ds.where(mask_center_iobs, drop=True)
        tbsim_edge[key] = ds.where(mask_edge_iobs, drop=True)

    return tbobs_center, tbobs_edge, tbsim_center, tbsim_edge

def plot_histograms(counts_all, counts_day,outfile):
    # --- Plot side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Define x-ticks for every 2 positions
    xticks = range(1, 31, 2)
    # Left: All observations
    axes[0].bar(counts_all.index.get_level_values(0), counts_all.values, color="skyblue", edgecolor="black")
    axes[0].set_xlabel("Scan position", fontsize=18)
    axes[0].set_ylabel("Number of observations", fontsize=18)
    axes[0].set_xticks(xticks)
    axes[0].tick_params(axis='both', labelsize=15)
    axes[0].set_title("a) Observations over 1 year\n      (single channel)", fontsize=18)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Right: Filtered day
    axes[1].bar(counts_day.index.get_level_values(0), counts_day.values, color="salmon", edgecolor="black")
    axes[1].set_xlabel("Scan position", fontsize=18)
    axes[1].set_ylabel("Number of observations", fontsize=18)
    axes[1].set_xticks(xticks)
    axes[1].tick_params(axis='both', labelsize=15)
    axes[1].set_title("b) Observations on 01/04/2024\n    (single channel)", fontsize=18)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Distribution of observations per scan position", fontsize=22)
    plt.tight_layout()
    # Save figure
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()


def plot_tb_diff_center_edge(tbobs_center, tbobs_edge, tbsim_center, tbsim_edge, title, outfile):
    """
    Plot Obs-Sim differences for center/edge scan positions and selected experiments.
    
    Rows:
        1: tbobs_center - tbsim_center['exp3']
        2: tbobs_center - tbsim_center['exp4']
        3: tbobs_edge   - tbsim_edge['exp3']
        4: tbobs_edge   - tbsim_edge['exp4']

    Columns: 4 channels
    """

    channel_names = ["23V", "31V", "50V", "53V"]
    lat_lims = [50, 90]
    n_rows = 4
    n_cols = 4
    exp_list = ['exp3', 'exp4']

    # --- Prepare figure ---
    fig, axes = sp_map(n_rows, n_cols, figsize=(10, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96, wspace=0.05, hspace=0.1)

    # --- Row labels ---
    row_labels = [
        "     Nadir\n Obs − Sim \n  exp3 [K]",
        "     Nadir\n Obs − Sim \n  exp4 [K]",
        "Large angles\n Obs − Sim \n   exp3 [K]",
        "Large angles\n Obs − Sim \n   exp4 [K]"
    ]

    # --- Loop over channels ---
    for ch in range(n_cols):
        # Select channel data
        obs_center_ch = tbobs_center.isel(channel=ch).tb
        obs_edge_ch   = tbobs_edge.isel(channel=ch).tb

        # Coordinates for plotting
        lon_center = tbobs_center.isel(channel=ch).lon
        lat_center = tbobs_center.isel(channel=ch).lat
        lon_edge   = tbobs_edge.isel(channel=ch).lon
        lat_edge   = tbobs_edge.isel(channel=ch).lat

        # Corresponding coordinates for each row
        lon_rows = [lon_center, lon_center, lon_edge, lon_edge]
        lat_rows = [lat_center, lat_center, lat_edge, lat_edge]

        # Compute differences for each row
        tb_diff_rows = [
            obs_center_ch - tbsim_center['exp3'].isel(channel=ch).tb,
            obs_center_ch - tbsim_center['exp4'].isel(channel=ch).tb,
            obs_edge_ch   - tbsim_edge['exp3'].isel(channel=ch).tb,
            obs_edge_ch   - tbsim_edge['exp4'].isel(channel=ch).tb
        ]

        # --- Determine common color limits for this channel ---
        vmin = min(diff.min().item() for diff in tb_diff_rows)
        vmax = max(diff.max().item() for diff in tb_diff_rows)
        zero_pos = (0 - vmin) / (vmax - vmin)
        cmap_diff = LinearSegmentedColormap.from_list(
            'blue_white_red',
            [(0.0, 'navy'), (zero_pos, 'white'), (1.0, 'darkred')]
        )

        # --- Plot each row ---
        for row, diff in enumerate(tb_diff_rows):
            ax = axes[row, ch]
            polarCentral_set_latlim(lat_lims, ax)

            im = ax.scatter(lon_rows[row],lat_rows[row],
                c=diff,
                s=2,
                cmap=cmap_diff,
                norm=Normalize(vmin=vmin, vmax=vmax),
                transform=ccrs.PlateCarree()
            )

            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.75, pad=0.03)
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            cbar.set_ticks([vmin, 0.0, vmax])
            cbar.ax.tick_params(labelsize=15)

            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
            ax.add_feature(cfeature.LAND, facecolor='gray')
            ax.gridlines()
            if row == 0:
                ax.set_title(channel_names[ch], fontsize=18, y=1.05)

    # --- Add row labels ---
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.45, 0.5, label, transform=axes[i, 0].transAxes,
                        rotation=90, va="center", fontsize=18)

    # --- Add figure title ---
    fig.suptitle(title, fontsize=22, y=1.03)

    # --- Add letters to each subplot ---
    letters = string.ascii_lowercase
    k = 0
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            ax.text(0.01, 1, f"{letters[k]})", transform=ax.transAxes,
                    fontsize=14, va="top", ha="left")
            k += 1

    # --- Save figure ---
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Figure saved as {outfile}")

def plot_sic_analysis_vs_ifs(
    lon_analysis,
    lat_analysis,
    sic_analysis,
    lon_ifs,
    lat_ifs,
    sic_ifs,
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
    sic_min = min(np.nanmin(sic_analysis), np.nanmin(sic_ifs))
    sic_max = max(np.nanmax(sic_analysis), np.nanmax(sic_ifs))

    norm_ice = Normalize(vmin=sic_min, vmax=sic_max)

    cmap_ice = LinearSegmentedColormap.from_list(
        'ice_cmap',
        [(0.0, 0.0, 0.3),
         (0.0, 0.2, 0.6),
         (0.5, 0.7, 1.0),
         (1.0, 1.0, 1.0)],
        N=100
    )

    sic_diff = sic_analysis - sic_ifs
    diff_min = np.nanmin(sic_diff)
    diff_max = np.nanmax(sic_diff)
    zero_pos = (0 - diff_min) / (diff_max - diff_min)

    cmap_diff = LinearSegmentedColormap.from_list(
        'blue_white_red',
        [(0.0, 'navy'), (zero_pos, 'white'), (1.0, 'darkred')]
    )

    norm_diff = Normalize(vmin=diff_min, vmax=diff_max)

    # ------------------------------------------------------------------
    # 3. Figure and axes
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        subplot_kw={'projection': ccrs.NorthPolarStereo()}
    )
    panel_labels = ['a)', 'b)', 'c)']  # Letters for subplots
    def _base_map(ax, title):
        ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
        ax.gridlines()
        ax.set_title(title, fontsize=24)

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    sc1 = axes[0].scatter(
        lon_analysis, lat_analysis,
        c=sic_analysis,
        s=point_size,
        cmap=cmap_ice,
        norm=norm_ice,
        transform=ccrs.PlateCarree()
    )
    _base_map(axes[0], f"Analysis\n{date_str}")
    axes[0].text(0.01, 0.95, panel_labels[0],
                transform=axes[0].transAxes,
                fontsize=28, va='top', ha='left')

    sc2 = axes[1].scatter(
        lon_ifs, lat_ifs,
        c=sic_ifs,
        s=point_size,
        cmap=cmap_ice,
        norm=norm_ice,
        transform=ccrs.PlateCarree()
    )
    _base_map(axes[1], f"IFS\n{date_str}")
    axes[1].text(0.01, 0.95, panel_labels[1],
                transform=axes[1].transAxes,
                fontsize=28, va='top', ha='left')

    sc3 = axes[2].scatter(
        lon_analysis, lat_analysis,
        c=sic_diff,
        s=point_size * 0.85,
        cmap=cmap_diff,
        norm=norm_diff,
        transform=ccrs.PlateCarree()
    )
    _base_map(axes[2], f"(Analysis − IFS)\n{date_str}")
    axes[2].text(0.01, 0.95, panel_labels[2],
                transform=axes[2].transAxes,
                fontsize=28, va='top', ha='left')

    # ------------------------------------------------------------------
    # 5. Colorbars
    # ------------------------------------------------------------------
    pos0, pos1, pos2 = (ax.get_position() for ax in axes)
    cbar_y = pos0.y0 - 0.10
    cbar_h = 0.025

    cax1 = fig.add_axes([
        pos0.x0,
        cbar_y,
        pos1.x1 - pos0.x0,
        cbar_h
    ])

    cb1 = fig.colorbar(sc1, cax=cax1, orientation='horizontal')
    cb1.set_label('Sea Ice Concentration', fontsize=22, labelpad=-65)
    cb1.ax.tick_params(labelsize=20)

    cax2 = fig.add_axes([
        pos2.x0,
        cbar_y,
        pos2.width,
        cbar_h
    ])

    cb2 = fig.colorbar(sc3, cax=cax2, orientation='horizontal')
    cb2.set_label('Sea Ice Concentration Difference', fontsize=22, labelpad=-65)
    cb2.ax.tick_params(labelsize=20)

    fig.savefig(f"{output_path}/SIC_01042024_exp4_without_nans"+'.png', dpi=300, bbox_inches="tight")


    return fig, axes


def plot_seaice_properties_consecutive_days(
    seaice_properties_ds,
    seaice_sic_ds,
    days,
    days_labels,
    output_path,
    lat_lims=(50, 90),
    figsize=(16, 20),
    point_size=20,
    sic_threshold=0.2
):
    """
    Plot two empirical sea-ice properties over several consecutive days,
    showing only points where sea-ice concentration exceeds sic_threshold.
    A single continuous colormap is used for all properties and days.
    Subplots are labeled a), b), ...
    """

    # ------------------------------------------------------------------
    # Extract coordinates
    # ------------------------------------------------------------------
    lon = seaice_properties_ds.lon.values
    lat = seaice_properties_ds.lat.values

    # ------------------------------------------------------------------
    # Apply SIC threshold mask
    # ------------------------------------------------------------------
    mask = seaice_sic_ds.seaice.isel(lagstep=days) > sic_threshold

    # ------------------------------------------------------------------
    # Compute global min/max across both properties and all days
    # ------------------------------------------------------------------
    data_masked = seaice_properties_ds.properties.isel(step=days).where(mask)
    global_min = float(data_masked.min())
    global_max = float(data_masked.max())

    norm = Normalize(vmin=global_min, vmax=global_max)

    # ------------------------------------------------------------------
    # Custom continuous colormap
    # ------------------------------------------------------------------
    colors = [
        (0.0, 0.0, 0.6), (0.0, 0.2, 0.8), (0.0, 0.4, 1.0), (0.2, 0.6, 1.0),
        (0.0, 0.6, 0.4), (0.0, 0.8, 0.3), (0.2, 0.9, 0.2), (0.6, 1.0, 0.2),
        (1.0, 1.0, 0.4), (1.0, 0.8, 0.2), (1.0, 0.6, 0.0), (1.0, 0.4, 0.0),
        (1.0, 0.2, 0.0), (0.9, 0.0, 0.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0)
    ]
    cmap = LinearSegmentedColormap.from_list("seaice_properties_cmap", colors, N=256)

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    nrows = len(days)
    fig, axes = plt.subplots(
        nrows, 2,
        figsize=figsize,
        subplot_kw={'projection': ccrs.NorthPolarStereo()}
    )

    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    # ------------------------------------------------------------------
    # Plot loop
    # ------------------------------------------------------------------
    for row, day in enumerate(days):
        for col in range(2):
            ax = axes[row, col]

            ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], crs=ccrs.PlateCarree())

            # Select masked data
            data = seaice_properties_ds.properties.isel(step=day, prop=col).where(
                seaice_sic_ds.seaice.isel(lagstep=day) > sic_threshold
            )

            sc = ax.scatter(
                lon,
                lat,
                c=data,
                s=point_size,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree()
            )

            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, facecolor='gray')
            ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
            ax.gridlines()

            ax.set_title(f"Property {col + 1} – {days_labels[row]}", fontsize=22)

            # Add panel label
            label_idx = row * 2 + col
            if label_idx < len(panel_labels):
                ax.text(
                    0.01, 1.05,
                    panel_labels[label_idx],
                    transform=ax.transAxes,
                    fontsize=18,
                    va='top',
                    ha='left'
                )

    # ------------------------------------------------------------------
    # Single shared colorbar for all subplots
    # ------------------------------------------------------------------
    pos0 = axes[-1, 0].get_position()
    pos1 = axes[-1, 1].get_position()
    cbar_height = 0.03
    cbar_y = pos0.y0 - 0.05
    cax = fig.add_axes([pos0.x0, cbar_y, pos1.x1 - pos0.x0, cbar_height])

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation='horizontal'
    )
    cb.ax.tick_params(labelsize=16)
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cb.set_label("Sea Ice Property Value", fontsize=18, labelpad=10)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    fig.savefig(f"{output_path}/seaice_properties_consecutive_days_exp4_single_colorbar.png",
                dpi=300, bbox_inches="tight")

    return fig, axes

def plot_properties_with_obs_column(
    seaice_properties_ds,
    seaice_sic_ds,
    properties_obs_ds,
    tbobs_ds,
    days,
    days_labels,       # ISO format YYYY-MM-DD for filtering
    days_titles,       # titles for subplots, e.g. "01/04/2024"
    output_path=".",
    lat_lims=(50, 90),
    figsize=(18, 20),
    point_size=20,
    sic_threshold=0.2
):
    """
    Plot two sea-ice properties on the grid for consecutive days, plus a third column
    with observation-based property values for the same days. Uses a common colorbar.
    """
    
    # ----------------------
    # 1. Extract coordinates
    # ----------------------
    lon_grid = seaice_properties_ds.lon.values
    lat_grid = seaice_properties_ds.lat.values
    
    # ----------------------
    # 2. Apply SIC mask to grid properties
    # ----------------------
    prop1_values_masked_list = []
    prop2_values_masked_list = []
    
    for day in days:
        sic = seaice_sic_ds.seaice.isel(lagstep=day).values
        mask = sic > sic_threshold
        prop1_values_masked_list.append(seaice_properties_ds.properties.isel(prop=0, step=day).values[mask])
        prop2_values_masked_list.append(seaice_properties_ds.properties.isel(prop=1, step=day).values[mask])
    
    # ----------------------
    # 3. Filter observations by day and store values
    # ----------------------
    prop_obs_masked_list = []
    for day_label in days_labels:
        target_date = np.datetime64(day_label)
        day_mask = tbobs_ds.date_time_fromjd.dt.floor('D') == target_date
        obs_indices = np.where(day_mask)[0]
        prop_obs_values = properties_obs_ds.properties.isel(iobs=obs_indices).values
        prop_obs_masked_list.append((prop_obs_values, obs_indices))
    
    # ----------------------
    # 4. Determine global min/max for normalization
    # ----------------------
    all_vals = np.concatenate(prop1_values_masked_list + prop2_values_masked_list +
                              [vals.ravel() for vals, idx in prop_obs_masked_list])
    global_min = np.nanmin(all_vals)
    global_max = np.nanmax(all_vals)
    
    norm = Normalize(vmin=global_min, vmax=global_max)
    
    # ----------------------
    # 5. Colormap
    # ----------------------
    colors = [
        (0.0, 0.0, 0.6), (0.0, 0.2, 0.8), (0.0, 0.4, 1.0), (0.2, 0.6, 1.0),
        (0.0, 0.6, 0.4), (0.0, 0.8, 0.3), (0.2, 0.9, 0.2), (0.6, 1.0, 0.2),
        (1.0, 1.0, 0.4), (1.0, 0.8, 0.2), (1.0, 0.6, 0.0), (1.0, 0.4, 0.0),
        (1.0, 0.2, 0.0), (0.9, 0.0, 0.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0)
    ]
    cmap = LinearSegmentedColormap.from_list("seaice_properties_cmap", colors, N=256)
    
    # ----------------------
    # 6. Create figure and axes
    # ----------------------
    nrows = len(days)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={'projection': ccrs.NorthPolarStereo()})
    
    letters = iter(ascii_lowercase)
    
    # ----------------------
    # 7. Plot loop
    # ----------------------
    for row, day in enumerate(days):
        for col in range(ncols):
            ax = axes[row, col]
            ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, facecolor='gray')
            ax.add_feature(cfeature.OCEAN, facecolor=(0.7,0.75,0.8))
            ax.gridlines()
            
            # Grid properties
            if col == 0:
                data = seaice_properties_ds.properties.isel(prop=0, step=day).values
                mask = seaice_sic_ds.seaice.isel(lagstep=day).values > sic_threshold
                data[~mask] = np.nan
                sc = ax.scatter(lon_grid, lat_grid, c=data, s=point_size, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            elif col == 1:
                data = seaice_properties_ds.properties.isel(prop=1, step=day).values
                mask = seaice_sic_ds.seaice.isel(lagstep=day).values > sic_threshold
                data[~mask] = np.nan
                sc = ax.scatter(lon_grid, lat_grid, c=data, s=point_size, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            # Observation column
            else:
                prop_obs_values, obs_indices = prop_obs_masked_list[day]
                data =  prop_obs_values
                lon = tbobs_ds.lon.isel(iobs=obs_indices).values
                lat = tbobs_ds.lat.isel(iobs=obs_indices).values
            
                sc = ax.scatter(lon, lat, c=data, s=point_size, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            
            # Title
            if col < 2:
                ax.set_title(f"Property {col+1} (grid space) \n{days_titles[row]}", fontsize=24)
            else:
                ax.set_title(f"Property 3 (obs. space) \n{days_titles[row]}", fontsize=24)
            
            # Subplot letter
            ax.text(0.01, 0.95, f"{next(letters)})", transform=ax.transAxes,
                    fontsize=22, va='top', ha='left')
    
    # ----------------------
    # 8. Shared colorbar for all subplots
    # ----------------------
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label("Property value", fontsize=24)
    cb.ax.tick_params(labelsize=22)
    
    # ----------------------
    # 9. Save figure
    # ----------------------
    fig.savefig(f"{output_path}/3seaice_properties.png", dpi=300, bbox_inches="tight")
    
    return fig, axes

def plot_ice_emis_all_channels_shared(tb_ds, days_labels,days_titles, output_path=".", figsize=(20, 15), point_size=20):
    """
    Plot ice emissivity for multiple channels over multiple days.
    Each row corresponds to a day, each column corresponds to a channel.
    All channels share the same color scale across all days.
    """
    n_days = len(days_labels)
    n_channels = len(tb_ds.channel)
    channel_names = ["23V", "31V", "50V", "53V"]

    # ----------------------
    # 1. Filter observations for each day and channel
    # ----------------------
    ice_emis_list = []  # list of lists: ice_emis_list[day][channel]
    lon_list = []
    lat_list = []

    for day_label in days_labels:
        target_date = np.datetime64(day_label)
        day_mask = tb_ds.date_time_fromjd.dt.floor('D') == target_date
        indices = np.where(day_mask)[0]

        lon_list.append(tb_ds.lon.isel(iobs=indices).values)
        lat_list.append(tb_ds.lat.isel(iobs=indices).values)

        channel_list = []
        for ch in tb_ds.channel.values:
            channel_list.append(tb_ds.ice_emis.isel(iobs=indices, channel=ch).values)
        ice_emis_list.append(channel_list)
    
    # ----------------------
    # 2. Determine global min/max across all channels and days
    # ----------------------
    all_vals = np.concatenate([ice_emis_list[day][ch].ravel()
                               for day in range(n_days) for ch in range(n_channels)])
    global_min = np.nanmin(all_vals)
    global_max = np.nanmax(all_vals)
    norm = Normalize(vmin=global_min, vmax=global_max)

    # ----------------------
    # 3. Colormap
    # ----------------------
    colors = [
        (0.0, 0.0, 0.6), (0.0, 0.2, 0.8), (0.0, 0.4, 1.0), (0.2, 0.6, 1.0),
        (0.0, 0.6, 0.4), (0.0, 0.8, 0.3), (0.2, 0.9, 0.2), (0.6, 1.0, 0.2),
        (1.0, 1.0, 0.4), (1.0, 0.8, 0.2), (1.0, 0.6, 0.0), (1.0, 0.4, 0.0),
        (1.0, 0.2, 0.0), (0.9, 0.0, 0.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0)
    ]
    cmap = LinearSegmentedColormap.from_list("ice_emis_cmap", colors, N=256)

    # ----------------------
    # 4. Create figure
    # ----------------------
    fig, axes = plt.subplots(n_days, n_channels, figsize=figsize, subplot_kw={'projection': ccrs.NorthPolarStereo()})
    letters = iter(ascii_lowercase)

    for i in range(n_days):
        for j in range(n_channels):
            ax = axes[i, j] if n_days > 1 else axes[j]
            ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, facecolor='gray')
            ax.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
            ax.gridlines()

            sc = ax.scatter(
                lon_list[i], lat_list[i],
                c=ice_emis_list[i][j],
                s=point_size,
                cmap=cmap,
                norm=norm,  # shared normalization for all channels
                transform=ccrs.PlateCarree()
            )

            ax.set_title(f"{days_titles[i]} – {channel_names[j]}", fontsize=24)
            ax.text(0.01, 0.95, f"{next(letters)})", transform=ax.transAxes,
                    fontsize=22, va='top', ha='left')

    # ----------------------
    # 5. Shared colorbar for all subplots
    # ----------------------
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label("Sea-ice emissivity", fontsize=24)
    cb.ax.tick_params(labelsize=22)

    # ----------------------
    # 6. Save figure
    # ----------------------
    fig.savefig(f"{output_path}/ice_emis_channels_shared_scale.png", dpi=300, bbox_inches="tight")

    return fig, axes


# ================= USER OPTIONS =================
DO_TABLES = False  # set True when you want tables
DO_PLOTS = False   # set True when you want plots
# =================================================

folder = "/perm/dnk8355/outputs_training_v2_jan26_report_final"
experiments = {
    "exp1": "bg_emis08_with_losses_original_obs_errors_bg_biasice2_5_ocean5_bg_bias_err0_001_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10",
    "exp2": "bg_emis08_with_losses_new_obs_errors_bg_biasice2_5_ocean5_bg_bias_err0_001_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10",
    "exp3": "bg_emis07_with_losses_new_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_02_newimplementation_in_emisNN_no_angle_sbatch_19jan_python3_10",
    "exp4": "bg_emis07_with_losses_new_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_02_newimplementation_in_emisNN_with_angle_sbatch_19jan_python3_10",
    "exp5": "bg_emis06_with_losses_new_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_02_newimplementation_in_emisNN_with_angle_sbatch_19jan_python3_10",
}
scanpos = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/METOP-B_SCANPOS.nc")
fg_dep = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/METOP-B_FG_DEP.nc")

tables = {}

target_date = np.datetime64("2024-04-01")
target_date_odb = np.datetime64("2024-04-01") 

for name, tag in experiments.items():
    print(f"\nProcessing {name}")
    # Create subfolder for plots
    exp_plot_dir = os.path.join(folder, name)
    os.makedirs(exp_plot_dir, exist_ok=True)

    # -------- TABLES --------
    if DO_TABLES:
        table = compute_tb_statistics(folder, tag, scanpos, fg_dep)
        tables[name] = table

        table_fmt = format_tb_statistics_table(table)

        save_tb_statistics_latex(
            table_fmt,
            exp_plot_dir,
            tag,
            name
        )

    # -------- PLOTS --------
    if DO_PLOTS:
        exp = load_experiment(folder, tag)

        tbobs_f, tbsim_f, tbsim_ini_f, fg_dep_f, scanpos_f = filter_by_day(
            exp["tbobs"],
            exp["tbsim"],
            exp["tbsim_ini"],
            fg_dep, scanpos,
            target_date,
            target_date_odb
        )
        outfile = os.path.join(
            exp_plot_dir,
            f"{str(name)}_tb_maps_{tag}_{str(target_date_odb)}.png"
        )
        plot_tb_maps(
            tbobs_f,
            tbsim_f,
            fg_dep_f,
            title=str(pd.to_datetime(target_date_odb).strftime("%d/%m/%Y")+' - Experiment '+str(name[3:])),
            outfile=outfile)
        

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

plot_tb_compare_experiments(
    tbobs_f,
    tbsim_all,
    title=str(pd.to_datetime(target_date_odb).strftime("%d/%m/%Y")),
    outfile=os.path.join(
            folder,
            f"BT_comparison_{str(target_date_odb)}.png"
        ))

plot_tb_diff_compare_experiments(
    tbobs_f,
    tbsim_all,
    fg_dep_f,
    title=str(pd.to_datetime(target_date_odb).strftime("%d/%m/%Y")),
    outfile=os.path.join(
            folder,
            f"BT_diff_comparison_{str(target_date_odb)}.png"
        ))


tbobs_f_center, tbobs_f_edge, tbsim_f_center, tbsim_f_edge=filter_by_scanpos(tbsim_all, tbobs_f, scanpos_f)


# Counts for all observations (left subplot)
counts_all = pd.DataFrame(scanpos["SCANPOS"]).value_counts().sort_index()

# Counts for filtered day (right subplot)
counts_day = pd.DataFrame(scanpos_f["SCANPOS"]).value_counts().sort_index()

plot_histograms(counts_all, counts_day,outfile=os.path.join(
            folder,
            f"histograms.png"
        ))

plot_tb_diff_center_edge(tbobs_f_center, tbobs_f_edge, tbsim_f_center, tbsim_f_edge, "01/04/2024", outfile=os.path.join(
            folder,"diff_nadir_edges_01042024.png"
        ))


#Extra analysis
# Plot of losses for first, fourth and fifth experiment
selected_exps = ['exp1', 'exp4', 'exp5']

models_all = {} 
for name in selected_exps:
    tag = experiments[name]
    exp = load_experiment(folder, tag)

    models_all[name] = exp['models']


plot_losses(models_all, selected_exps, output_path=folder)


#Plot of SIC from hybrid ML model, IFS and differences for 1 day
#For this case we only use experiment 4
exp4 = load_experiment(folder, experiments['exp4'])
seaice_exp4=exp4['seaice']
#For the moment we transform negative values in nans
seaice_exp4 = seaice_exp4.where(seaice_exp4.seaice >= 0)

# Open daily sea ice from IFS
daily_ifs_sic_without_land_without_nans = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/ifs_seaice_METOP-B_1apr2024_31march2025_daily_without_land_without_nans.nc")
#daily_ifs_sic_without_land = xr.open_dataset("/perm/dnk8355/netcdf_1april2024_31march2025/ifs_seaice_METOP-B_1apr2024_31march2025_daily_without_land.nc")


def day_index(date_str, start_str="2024-04-01"):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start = datetime.strptime(start_str, "%Y-%m-%d")
    return (date - start).days 

day=day_index("2024-04-01")


# --- First dataset: analysis ---
seaice_analysis = seaice_exp4.seaice.isel(lagstep=day).values
lon_analysis = seaice_exp4.lon.isel(lagstep=day).values
lat_analysis = seaice_exp4.lat.isel(lagstep=day).values

# --- Second dataset: IFS initial conditions ---
seaice_ifs = daily_ifs_sic_without_land_without_nans.SIC[:, day].values
lon2 = daily_ifs_sic_without_land_without_nans.LON.values
lat2 = daily_ifs_sic_without_land_without_nans.LAT.values


fig, axes = plot_sic_analysis_vs_ifs(
    lon_analysis,
    lat_analysis,
    seaice_analysis,
    lon2,
    lat2,
    seaice_ifs,
    date_str="01/04/2024",output_path=folder,point_size=45,
)


#################################################
#Plot properties 1 and 2 over 3 consecutive days
###################################################
#For this case we only use experiment 4
def day_index(date_str, start_str="2024-04-01"):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start = datetime.strptime(start_str, "%Y-%m-%d")
    return (date - start).days

# Example: plot 3 consecutive days starting 2024-04-01
day0 = day_index("2024-04-01")
days = [day0, day0+1, day0+2]  # lagstep indices in the grid
days_labels = ["2024-04-01", "2024-04-02", "2024-04-03"]  # ISO format for filtering tbobs
days_titles = ["01/04/2024", "02/04/2024", "03/04/2024"]  # for subplot titles

exp4 = load_experiment(folder, experiments['exp4'])

fig, axes = plot_properties_with_obs_column(
    seaice_properties_ds=exp4['properties_grid'],
    seaice_sic_ds=exp4['seaice'],
    properties_obs_ds=exp4['properties_obs'],
    tbobs_ds=exp4['tbobs'] ,
    days=days,
    days_labels=days_labels,
    days_titles=days_titles,
    output_path=folder,     # folder to save the figure
    lat_lims=(50, 90),
    figsize=(18, 20),
    point_size=20,
    sic_threshold=0.2       # only show grid cells with SIC > 0.2
)

#Plot emissivity for all channels and 3 consecutive days
days_labels = ["2024-04-01", "2024-04-02", "2024-04-03"]

fig, axes = plot_ice_emis_all_channels_shared(
    tb_ds=exp4['ice_emis'],
    days_labels=days_labels,days_titles=days_titles,
    output_path=folder,
    figsize=(20, 15),
    point_size=20
)



#Plot maps and emissivity according to channel for 1 day and 1 location
# -----------------------------
# Parámetros
# -----------------------------
day_label = "2024-04-01"
target_date = np.datetime64(day_label)

points = {
    "perpetual_ice": {
        "coord": (-58, 86),
        "color": "tab:blue",
        "title": "Multi-year Ice"
    },
    "seasonal_ice": {
        "coord": (79, 76),
        "color": "tab:orange",
        "title": "Seasonal Ice"
    }
}

channel_labels = ["23V", "31V", "50V", "53V"]

tb = exp4["ice_emis"]   # alias corto
letters = iter(ascii_lowercase)

# -----------------------------
# Crear figura
# -----------------------------
fig = plt.figure(figsize=(12, 10))

# =============================
# a) MAPA (fila superior)
# =============================
ax_map = plt.subplot2grid((2, 4), (0, 1), colspan=2,
                          projection=ccrs.NorthPolarStereo())

polarCentral_set_latlim([50, 90], ax_map)

ax_map.add_feature(cfeature.COASTLINE)
ax_map.add_feature(cfeature.LAND, facecolor="gray")
ax_map.add_feature(cfeature.OCEAN, facecolor=(0.7, 0.75, 0.8))
ax_map.gridlines()

for p in points.values():
    lon_pt, lat_pt = p["coord"]
    ax_map.plot(
        lon_pt, lat_pt, "o",
        color=p["color"],
        markersize=8,
        transform=ccrs.PlateCarree()
    )

ax_map.set_title("Study points in the Arctic", fontsize=22)
ax_map.text(-0.18, 1.02, "a)",
            transform=ax_map.transAxes,
            fontsize=22)

# =============================
# b) y c) EMISIVIDAD vs CANAL
# =============================

# Filtrar observaciones del día
mask = tb.date_time_fromjd.dt.floor("D") == target_date
indices = np.where(mask)[0]

# Subplots normales (NO Cartopy)
ax_b = plt.subplot2grid((2, 4), (1, 0), colspan=2)
ax_c = plt.subplot2grid((2, 4), (1, 2), colspan=2)

for ax, key, label in zip(
        [ax_b, ax_c],
        points.keys(),
        ["b)", "c)"]):

    lon_pt, lat_pt = points[key]["coord"]
    color = points[key]["color"]

    # Distancia al punto (usar .values para evitar problemas con xarray)
    dist2 = (
        (tb.lon.isel(iobs=indices).values - lon_pt) ** 2 +
        (tb.lat.isel(iobs=indices).values - lat_pt) ** 2
    )

    iobs = indices[np.argmin(dist2)]
    values = tb.ice_emis.isel(iobs=iobs).values

    ax.plot(
        np.arange(1, values.size + 1),
        values,
        "o-",
        color=color
    )

    ax.set_xticks(np.arange(1, values.size + 1))
    ax.set_xticklabels(channel_labels, fontsize=22)

    ax.set_ylim(0.8, 1)
    ax.set_xlabel("Channel", fontsize=22)
    ax.set_ylabel("Sea-ice emissivity", fontsize=22)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18, right=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Minor ticks en Y
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # 1 minor tick entre majors

    # Apariencia de los ticks
    ax.tick_params(axis='y', which='minor', length=2, right=True)
    ax.tick_params(axis='y', which='major', length=5)

    ax.set_title(points[key]["title"], fontsize=22)

    ax.text(0.03, 1.015, label,
            transform=ax.transAxes,
            fontsize=22)

# -----------------------------
# Ajuste final
# -----------------------------
plt.tight_layout()
plt.subplots_adjust(wspace=0.99)
plt.show()
fig.savefig(
    f"{folder}/ice_emis_points_channels_{day_label}.png",
    dpi=300,
    bbox_inches="tight"
)





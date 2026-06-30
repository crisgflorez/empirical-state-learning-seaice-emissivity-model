import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import os
import matplotlib.lines as mlines

# =========================================================
# 1. CONFIG
# =========================================================

file_path = "/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/daily_by_hemisphere_above_below_45.pkl"
#file_path = "/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/daily_global_above_below_45.pkl"

name = os.path.basename(file_path).replace("daily_", "").replace(".pkl", "")

output_path_tsfc_wind = (
    "/home/dnk8355/EUMETSAT_fellowship/"
    "empirical-state-learning-seaice-emissivity-model/plots/"
    f"tsfc_windspeed_timeseries_{name}.png"
)

variables = ["TSFC", "WINDSPEED10M"]

var_labels = {
    "TSFC": "Model skin temperature (K)",
    "WINDSPEED10M": "10 m wind speed (m/s)"
}

date_col = "day"
sat_col = "satellite"

# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_pickle(file_path)
df[date_col] = pd.to_datetime(df[date_col])

has_hemisphere = "hemisphere" in df.columns

satellites = sorted(df[sat_col].unique())

palette = ['#117733', '#88CCEE', '#DDCC77', '#CC6677', '#332288']
sat_colors = dict(zip(satellites, palette[:len(satellites)]))

if has_hemisphere:
    hemispheres = sorted(df["hemisphere"].unique())
else:
    hemispheres = ["Global"]

other_vars = sorted(df["var"].unique())
other_vars = [v for v in other_vars if v not in variables]





# =========================================================
# 3. FIRST FIGURE
# =========================================================

nrows = len(variables)
ncols = len(hemispheres)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(8 * ncols, 5 * nrows),
    sharex=True,
    sharey='row'
)

if ncols == 1:
    axes = [[ax] for ax in axes]


# Hemisphere pretty labels
hemi_labels = {
    "Global": "Global (>45 & <-45 degrees)",
    "North": "North (>45 degrees)",
    "South": "South (<-45 degrees)"
}

for i, var in enumerate(variables):

    for j, hemi in enumerate(hemispheres):

        ax = axes[i][j]

        for sat in satellites:

            if has_hemisphere:
                sub = df[
                    (df[sat_col] == sat) &
                    (df["hemisphere"] == hemi) &
                    (df["var"] == var)
                ]
            else:
                sub = df[
                    (df[sat_col] == sat) &
                    (df["var"] == var)
                ]

            if sub.empty:
                continue

            sub = sub.sort_values(date_col)
            sub = sub.dropna(subset=["min", "max", "mean"])

            ax.plot(
                sub[date_col],
                sub["min"],
                linestyle="dashed",
                linewidth=2,
                color=sat_colors[sat],
                label=f"{sat} min"
            )

            ax.plot(
                sub[date_col],
                sub["max"],
                linestyle="solid",
                linewidth=2,
                color=sat_colors[sat],
                label=f"{sat} max"
            )

            ax.plot(
                sub[date_col],
                sub["mean"],
                linestyle="dotted",
                linewidth=2,
                color=sat_colors[sat],
                label=f"{sat} mean"
            )


        # TITLES
        ax.set_title(
            hemi_labels.get(hemi, hemi),
            fontsize=18
        )

        # LABELS

        if j == 0:
            ax.set_ylabel(
                var_labels.get(var, var),
                fontsize=16
            )


        # TICKS STYLE
        ax.tick_params(axis='both', labelsize=14)


        # GRID
        ax.grid(True, alpha=0.3)

        # X AXIS: MONTHLY TICKS + LABEL EVERY 2 MONTHS
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, pos: mdates.num2date(x).strftime('%m/%y')
                if mdates.num2date(x).month % 2 == 1 else ''
            )
        )
        ax.tick_params(axis='x', labelrotation=30)


#LEGEND
handles, labels = axes[0][0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=6,
    frameon=True
)
plt.tight_layout(rect=[0, 0.02, 1, 1])

plt.savefig(
    output_path_tsfc_wind,
    dpi=300,
    bbox_inches="tight"
)



# =========================================================
# 4. FIGURE LOOP (OTHER VARIABLES + HEMISPHERE SAFE)
# =========================================================
# OUTPUT PATH BASE
base_name = os.path.basename(file_path).replace("daily_", "").replace(".pkl", "")

output_dir = "/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/plots"

satellites_to_plot = ["METOP-B", "METOP-C"]

sat_colors = {
    "METOP-B": "#117733",
    "METOP-C": "#88CCEE"
}

has_hemisphere = "hemisphere" in df.columns

if has_hemisphere:
    hemispheres = sorted(df["hemisphere"].unique())
    name_out= 'by_hemisphere'
else:
    hemispheres = ["Global"]
    name_out= 'global'

for var in other_vars:

    sub = df[df["var"] == var]

    channels = sorted(sub["channel"].dropna().unique())
    if len(channels) == 0:
        continue

    nrows = len(channels)

    # columns = satellites × hemispheres
    ncols = len(satellites_to_plot) * len(hemispheres)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        sharex=True,
        squeeze=False
    )

    for i, ch in enumerate(channels):

        for j, sat in enumerate(satellites_to_plot):

            for k, hemi in enumerate(hemispheres):

                col = j * len(hemispheres) + k
                ax = axes[i][col]

                if has_hemisphere:
                    sat_data = sub[
                        (sub["channel"] == ch) &
                        (sub["satellite"] == sat) &
                        (sub["hemisphere"] == hemi)
                    ]
                else:
                    sat_data = sub[
                        (sub["channel"] == ch) &
                        (sub["satellite"] == sat)
                    ]

                if sat_data.empty:
                    ax.set_visible(False)
                    continue

                sat_data = sat_data.sort_values("day")

                # =========================
                # MIN
                # =========================
                ax.plot(
                    sat_data["day"],
                    sat_data["min"],
                    linestyle="--",
                    linewidth=2,
                    color=sat_colors.get(sat, "black"),
                    label="min"
                )

                # =========================
                # MAX
                # =========================
                ax.plot(
                    sat_data["day"],
                    sat_data["max"],
                    linestyle="-",
                    linewidth=2,
                    color=sat_colors.get(sat, "black"),
                    label="max"
                )

                # =========================
                # MEAN
                # =========================
                ax.plot(
                    sat_data["day"],
                    sat_data["mean"],
                    linestyle=":",
                    linewidth=2,
                    color=sat_colors.get(sat, "black"),
                    label="mean"
                )

                # =========================
                # TITLE
                # =========================
                title = f"{var} | Ch {int(ch)} | {sat}"
                if has_hemisphere:
                    title += f" | {hemi}"

                ax.set_title(title, fontsize=20)

                ax.grid(alpha=0.3)
                ax.tick_params(axis='both', labelsize=20)

                # =========================
                # X AXIS
                # =========================
                ax.xaxis.set_major_locator(mdates.MonthLocator())

                ax.xaxis.set_major_formatter(
                    FuncFormatter(
                        lambda x, pos: (
                            mdates.num2date(x).strftime("%m/%y")
                            if mdates.num2date(x).month in [1, 4, 7, 10]
                            else ""
                        )
                    )
                )

                ax.tick_params(axis="x", labelrotation=30)

                if i == nrows - 1:
                    ax.set_xlabel("Date", fontsize=20)

                if (j == 0) and (k == 0):
                    ax.set_ylabel("Value", fontsize=20)
    line_min = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='min')
    line_max = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='max')
    line_mean = mlines.Line2D([], [], color='black', linestyle=':', linewidth=2, label='mean')

    fig.legend(
        handles=[line_min, line_max, line_mean],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.1),
        ncol=3,
        frameon=True,
        fontsize=20
    )
    fig.tight_layout()

    # =========================================================
    # SAVE FIGURE
    # =========================================================
    output_file = os.path.join(
        output_dir,
        f"{var}_per_channel_{name_out}_above_below_45.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)





# =========================================================
# 5. LAST FIGURE (COUNTING)
# =========================================================

nrows = len(variables)
ncols = len(hemispheres)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(8 * ncols, 5 * nrows),
    sharex=True,
    sharey='row'
)

if ncols == 1:
    axes = [[ax] for ax in axes]

# Hemisphere pretty labels
hemi_labels = {
    "Global": "Global (>45 & <-45 degrees)",
    "North": "North (>45 degrees)",
    "South": "South (<-45 degrees)"
}

for i, var in enumerate(variables):

    for j, hemi in enumerate(hemispheres):

        ax = axes[i][j]

        for sat in satellites:

            if has_hemisphere:
                sub = df[
                    (df[sat_col] == sat) &
                    (df["hemisphere"] == hemi) &
                    (df["var"] == var)
                ]
            else:
                sub = df[
                    (df[sat_col] == sat) &
                    (df["var"] == var)
                ]

            if sub.empty:
                continue

            sub = sub.sort_values(date_col)

            ax.plot(
                sub[date_col],
                sub["count"],
                linewidth=2,
                color=sat_colors[sat],
                label=sat
            )


        # TITLE
        ax.set_title(
            hemi_labels.get(hemi, hemi),
            fontsize=18
        )

        # LABELS
        if j == 0:
            ax.set_ylabel("Number of observations", fontsize=16)

        ax.tick_params(axis='both', labelsize=14)
        ax.grid(alpha=0.3)

        # X AXIS FORMAT
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, pos: (
                    mdates.num2date(x).strftime("%m/%y")
                    if mdates.num2date(x).month in [1, 4, 7, 10]
                    else ""
                )
            )
        )

        ax.tick_params(axis="x", labelrotation=30)


#LEGEND
handles, labels = axes[0][0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=6,
    frameon=True,
    fontsize=14
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

# SAVE
output_file = os.path.join(
    output_dir,
    f"count_timeseries_{name}.png"
)

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

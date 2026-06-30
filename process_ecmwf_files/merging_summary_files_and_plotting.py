import os
import glob
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================================================
# 1. INPUT DIRECTORY
# =========================================================
txt_dir_daily = "/perm/dnk8355/paper2026/netcdf_daily_1april2024_31march2026"
txt_dir_all = "/perm/dnk8355/paper2026/netcdf_1april2024_31march2026"

# =========================================================
#    OUTPUT DIRECTORY
# =========================================================
dir_plot = "/home/dnk8355/EUMETSAT_fellowship/empirical-state-learning-seaice-emissivity-model/plots"

# =========================================================
# 2. CHECK IF ALL SUMMARY FILES ALREADY EXIST
# =========================================================
all_summary_files = glob.glob(os.path.join(txt_dir_all, "*_all_summary.txt"))

skip_merge = len(all_summary_files) > 0

if skip_merge:
    print("[INFO] ALL_SUMMARY files already exist → skipping merge")
else:
    print("[INFO] Creating ALL_SUMMARY files")

    files = glob.glob(os.path.join(txt_dir_daily, "*_summary.txt"))

    satellite_files = defaultdict(list)

    for f in files:
        name = os.path.basename(f)

        sat = "_".join(name.split("_")[:-2])
        satellite_files[sat].append(f)

    for sat, flist in satellite_files.items():

        flist = sorted(flist)

        output_file = os.path.join(
            txt_dir_all,
            f"{sat}_all_summary.txt"
        )

        with open(output_file, "w") as fout:

            for i, infile in enumerate(flist):

                with open(infile, "r") as fin:
                    fout.write(fin.read())

                if i < len(flist) - 1:
                    fout.write("\n" + "=" * 80 + "\n\n")

        print(f"[CREATED] {output_file}")

# =========================================================
# 3. READ ALL SUMMARY FILES → DATAFRAME
# =========================================================
def read_summary_file(filename):

    rows = []

    with open(filename) as f:
        content = f.read()

    blocks = content.split("=" * 80)

    for block in blocks:

        if "Satellite:" not in block:
            continue

        lines = [l.strip() for l in block.splitlines() if l.strip()]

        sat = lines[0].split(":")[1].split(",")[0].strip()
        date = pd.to_datetime(lines[1].split(":")[1].strip())

        total_obs = int(re.search(r'(\d+)', lines[3]).group(1))

        for line in lines[5:]:

            m = re.search(r'Channel\s+([\d\.]+):\s+(\d+)', line)

            if m:

                ch = int(float(m.group(1)))
                n_invalid = int(m.group(2))

                rows.append({
                    "satellite": sat,
                    "date": date,
                    "channel": ch,
                    "invalid_obs": n_invalid,
                    "total_obs": total_obs,
                    "percentage_invalid": 100 * n_invalid / total_obs
                })

    return pd.DataFrame(rows)

# =========================================================
# 4. BUILD DATAFRAME
# =========================================================
df_list = [read_summary_file(f) for f in glob.glob(os.path.join(txt_dir_all, "*_all_summary.txt"))]
df = pd.concat(df_list, ignore_index=True)

# =========================================================
# 5. PLOT
# =========================================================
satellites = sorted(df["satellite"].unique())
channels = sorted(df["channel"].unique())

palette = ['#6A4C93', '#E69F00', '#D55E00', '#009E73', '#CC79A7']
channel_colors = dict(zip(channels, palette))

fig, axes = plt.subplots(
    nrows=len(satellites),
    ncols=1,
    figsize=(18, 5 * len(satellites)),
    sharex=True
)

if len(satellites) == 1:
    axes = [axes]

y_lims_left = [[] for _ in range(len(satellites))]
y_lims_right = [[] for _ in range(len(satellites))]

for i, sat in enumerate(satellites):

    ax = axes[i]
    sat_df = df[df["satellite"] == sat]

    use_dual_axis = (sat == "METOP-B")
    ax2 = ax.twinx() if use_dual_axis else None

    for ch in channels:

        ch_df = sat_df[sat_df["channel"] == ch]

        if ch_df.empty:
            continue

        color = channel_colors.get(ch, "black")

        if ch == 15 and use_dual_axis:

            ax2.plot(
                ch_df["date"],
                ch_df["percentage_invalid"],
                color=color,
                linewidth=3,
                label="Ch 15"
            )

        else:

            ax.plot(
                ch_df["date"],
                ch_df["percentage_invalid"],
                color=color,
                linewidth=3,
                label=f"Ch {ch}"
            )
    ymin, ymax = ax.get_ylim()
    y_lims_left[i].extend([ymin, ymax])
    ax.set_title(sat, fontsize=23)
    ax.set_ylabel("% invalid obs", fontsize=23)
    ax.grid(True)

    if use_dual_axis:
        ax2.set_ylabel("Ch 15 % invalid",
                       fontsize=23,
                       color="#CC79A7")
        ax2.tick_params(axis='y', colors="#CC79A7",labelsize=22)
        ax2.spines['right'].set_color("#CC79A7")
        ax2.set_ylim(0,105)

# =========================================================
# 6. DATE FORMATTING
# =========================================================
major = mdates.MonthLocator(interval=1)
minor = mdates.MonthLocator(bymonthday=15)
fmt = mdates.DateFormatter("%m/%Y")

for ax in axes:
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_locator(minor)
    ax.tick_params(axis='x', rotation=45,labelsize=18)
    ax.grid(which='major', linestyle='-', linewidth=1, color='gray', alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.7, color='gray', alpha=0.5)
    ax.tick_params(axis='y', labelsize=22)    

# =========================================================
# 7. LEGEND
# =========================================================
# Legend from last axis
lines, labels = axes[-1].get_legend_handles_labels()

fig.legend(
    lines,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5,-0.04), 
    fontsize=22,
    ncol=5,
    columnspacing=0.5,
    frameon=True,
    handlelength=2
)

# =========================================================
# 8. SAVE + SHOW
# =========================================================
plt.suptitle("Percentage of invalid observations by channel", fontsize=25)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.savefig(
    os.path.join(dir_plot, "time_series_invalid_obs_per_channel.png"),
    bbox_inches="tight"
)

plt.show()
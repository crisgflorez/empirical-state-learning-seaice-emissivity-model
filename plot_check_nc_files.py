import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_pickle(
    "/home/dnk8355/perm/paper2026/netcdf_1april2024_31march2026/daily_statistics_above_below50degrees.pkl"
)

df["day"] = pd.to_datetime(df["day"])

# =========================================================
# 2. RECONSTRUCT SATELLITE FROM FILE NAME
# =========================================================
df["satellite"] = df["file"].str.split("_").str[0]

sat_colors = {
    "METOP-B": "#117733",
    "METOP-C": "#88CCEE"
}

# =========================================================
# 3. VARIABLES
# =========================================================
basic_vars = ["TSFC", "WINDSPEED10M"]

# =========================================================
# 4. OPTIONAL: HEMISPHERE MODE or both hemispheres together
# =========================================================
USE_HEMISPHERE = False   # <-- switch here

if USE_HEMISPHERE:
    hemispheres = ["North", "South"]
else:
    hemispheres = [None]

# =========================================================
# 5. AGGREGATE GLOBAL (from hemisphere split if present)
# =========================================================
group_cols = ["day", "satellite", "var"]

if "hemisphere" in df.columns and USE_HEMISPHERE:

    df_global = (
        df.groupby(group_cols + ["hemisphere"], as_index=False)
          .agg(
              min=("min", "min"),
              max=("max", "max"),
              mean=("mean", "mean"),
              count=("count", "sum")
          )
    )
else:
    df_global = (
        df.groupby(group_cols, as_index=False)
          .agg(
              min=("min", "min"),
              max=("max", "max"),
              mean=("mean", "mean"),
              count=("count", "sum")
          )
    )


# =========================================================
# 6. FIGURE 1: BASIC VARIABLES
# =========================================================
fig, axes = plt.subplots(
    len(basic_vars),
    len(hemispheres),
    figsize=(16, 10),
    sharex=True,
    squeeze=False
)

for j, hemi in enumerate(hemispheres):

    for i, var in enumerate(basic_vars):

        ax = axes[i][j]

        sub = df_global[df_global["var"] == var]

        if hemi is not None:
            sub = sub[sub["hemisphere"] == hemi]

        for sat in ["METOP-B", "METOP-C"]:

            sat_data = sub[sub["satellite"] == sat]

            ax.plot(
                sat_data["day"],
                sat_data["min"],
                linestyle="--",
                color=sat_colors[sat],
                label=f"{sat} min"
            )

            ax.plot(
                sat_data["day"],
                sat_data["max"],
                linestyle="-",
                color=sat_colors[sat],
                label=f"{sat} max"
            )

        title = var if hemi is None else f"{var} - {hemi}"
        ax.set_title(title)

        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        if j == 0:
            ax.set_ylabel("Value")

# =========================================================
# 7. LEGEND + SAVE
# =========================================================
fig.legend(loc="lower center", ncol=4)
plt.tight_layout(rect=[0, 0.05, 1, 1])












# =========================================================
# 2. DEFINE GROUPS
# =========================================================
basic_vars = ["TSFC", "WINDSPEED10M"]

other_vars = [
    "OBSVALUE","EMIS_WATER","TAUSFC","TDOWN","TUP",
    "TAUSFC_CLD","TDOWN_CLD","TUP_CLD","FG_DEP","AN_DEP"
]

sat_colors = {
    "METOP-B": "#117733",
    "METOP-C": "#88CCEE"
}

# =========================================================
# 3. FILTER HELPERS
# =========================================================
def sat_from_file(f):
    if "METOP-B" in f:
        return "METOP-B"
    elif "METOP-C" in f:
        return "METOP-C"
    return "UNKNOWN"


df["satellite"] = df["file"].apply(sat_from_file)

group_cols = ["day", "satellite", "var"]
df_global = (
    df.groupby(group_cols, as_index=False)
      .agg(
          min=("min", "min"),
          max=("max", "max"),
          mean=("mean", "mean"),
          count=("count", "sum")
      )
)


# =========================================================
# 4. FIGURE 1: BASIC VARIABLES (2 subplots)
# =========================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i, var in enumerate(basic_vars):

    ax = axes[i]
    sub = df_global[df_global["var"] == var]

    for sat in ["METOP-B", "METOP-C"]:

        sat_data = sub[sub["satellite"] == sat]

        ax.plot(
            sat_data["day"],
            sat_data["min"],
            linestyle="--",
            color=sat_colors[sat],
            label=f"{sat} min"
        )

        ax.plot(
            sat_data["day"],
            sat_data["max"],
            linestyle="-",
            color=sat_colors[sat],
            label=f"{sat} max"
        )

    ax.set_title(var)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

axes[0].set_ylabel("Value")
fig.legend(loc="lower center", ncol=4)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig("basic_variables_min_max.png", dpi=200)
plt.show()

# =========================================================
# 5. FIGURE 2: OTHER VARIABLES (subplots per channel)
# =========================================================
for var in other_vars:

    sub = df[df["var"] == var]

    channels = sorted(sub["channel"].dropna().unique())

    ncols = 2
    nrows = len(channels)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows), sharex=True, squeeze=False)

    for i, ch in enumerate(channels):

        for j, sat in enumerate(["METOP-B", "METOP-C"]):

            ax = axes[i][j]
            sat_data = sub[(sub["channel"] == ch) & (sub["satellite"] == sat)]

            if sat_data.empty:
                ax.set_visible(False)
                continue

            ax.plot(
                sat_data["day"],
                sat_data["min"],
                linestyle="--",
                color=sat_colors[sat],
                label="min"
            )

            ax.plot(
                sat_data["day"],
                sat_data["max"],
                linestyle="-",
                color=sat_colors[sat],
                label="max"
            )

            ax.set_title(f"{var} | Channel {int(ch)} | {sat}")
            ax.grid(alpha=0.3)

            if i == nrows - 1:
                ax.set_xlabel("Date")

            if j == 0:
                ax.set_ylabel("Value")

    fig.tight_layout()
    plt.savefig(f"{var}_per_channel_min_max.png", dpi=200)
    plt.show()
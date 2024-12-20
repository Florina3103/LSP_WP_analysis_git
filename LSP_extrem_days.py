# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec


params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.title_fontsize": "x-large",
    "lines.linewidth": 4,
    "lines.markersize": 10,
    "hatch.linewidth": 0.1,
}
plt.rcParams.update(params)

# %% define functions
def count_frequency(arr):
    frequency_dict = {}

    for element in arr:
        if element in frequency_dict:
            frequency_dict[element] += 1
        else:
            frequency_dict[element] = 1

    return frequency_dict


def count_frequency_given_keys(arr, keys):
    frequency_dict = {key: 0 for key in keys}  # Initialize all keys with 0

    for element in arr:
        if element in frequency_dict:
            frequency_dict[element] += 1

    return frequency_dict


def dic_percentage(dic, len):
    percentage_dict = {}

    for key in dic.keys():
        percentage_dict[key] = np.round(dic[key] / len * 100, 1)

    return percentage_dict


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter DJF"
    elif month in [3, 4, 5]:
        return "Spring MAM"
    elif month in [6, 7, 8]:
        return "Summer JJA"
    else:
        return "Autumn SON"


def select_data_between_years(df, start_year, end_year):
    return df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]


params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "hatch.linewidth": 0.1,
}
plt.rcParams.update(params)

pal = sns.color_palette("colorblind")
pal_hex = pal.as_hex()
# %% read in data

station = "WEG_L"
home_path = os.getcwd()

# Read 20CRv3 data
df1 = pd.read_csv(f"/home/flo/LSP_analysis/Data/AT_20CRv3_{station}_daily.csv")

# Read SOM data
df2 = pd.read_csv(
    home_path
    + "/Data/SOM_8_ssim_hgt_GRl_1900_2015/bmu_SOM_8_ssim_hgt_GRl_1900_2015.csv"
)

# Convert the 'date' column to datetime format
df1["date"] = pd.to_datetime(df1["time"])
df2["date"] = pd.to_datetime(df2["time"])

# Merge the two DataFrames on the 'date' column
merged_df = pd.merge(df1, df2, on="date", how="inner")
merged_df = merged_df[["date", "AT", "bmu"]]

# %%
merged_df["month"] = merged_df["date"].dt.month

# Map months to seasons using the function
merged_df["season"] = merged_df["month"].apply(get_season)
merged_df["DOY"] = merged_df["date"].dt.dayofyear

# Calculate the anomaly of 'AT' based on the average temperature in each season
seasonal_mean = merged_df.groupby("season")["AT"].mean()
mean_by_month = merged_df.groupby("month")["AT"].mean()
mean_by_day = merged_df.groupby("DOY")["AT"].mean()
merged_df["AT_rolling"] = merged_df["AT"].rolling(30, center=True).mean()
mean_by_day_rolling = merged_df.groupby("DOY")["AT_rolling"].mean()


merged_df["AT_anomaly"] = merged_df.apply(
    lambda row: row["AT"] - seasonal_mean[row["season"]], axis=1
)
merged_df["AT_anomaly_month"] = merged_df.apply(
    lambda row: row["AT"] - mean_by_month[row["month"]], axis=1
)
merged_df["AT_anomaly_doy"] = merged_df.apply(
    lambda row: row["AT"] - mean_by_day[row["DOY"]], axis=1
)
merged_df["AT_anomaly_rolling"] = merged_df.apply(
    lambda row: row["AT"] - mean_by_day_rolling[row["DOY"]], axis=1
)

merged_df["Period"] = "1900-2015"

merged_df.loc[
    (merged_df["date"] >= "1922-01-01") & (merged_df["date"] <= "1932-12-31"), "Period"
] = "WP1 1922-1932"
merged_df.loc[
    (merged_df["date"] >= "1993-01-01") & (merged_df["date"] <= "2007-12-31"), "Period"
] = "WP2 1993-2007"

df1 = select_data_between_years(merged_df, 1922, 1932)
df1["WP index"] = "WP1"
df1_freq = count_frequency(df1["bmu"])
df1_perc = dic_percentage(df1_freq, len(df1["bmu"]))

df2 = select_data_between_years(merged_df, 1993, 2007)
df2["WP index"] = "WP2"
df2_freq = count_frequency(df2["bmu"])
df2_perc = dic_percentage(df2_freq, len(df2["bmu"]))


df_freq = count_frequency(merged_df["bmu"])
df_perc = dic_percentage(df_freq, len(merged_df["bmu"]))

# %% LSP Manuscript Figure 6
percentage = 15
temp = merged_df
sort_value = "AT_anomaly_rolling"



fig = plt.figure(figsize=(15, 8), constrained_layout=True)
gs = GridSpec(2, 4, height_ratios=[7, 1], width_ratios=[1, 1, 1, 1], figure=fig)

# Subplot 1: Horizontal Bar Plot (on the left)
ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[0, 1])
cx = fig.add_subplot(gs[0, 2])
dx = fig.add_subplot(gs[0, 3])

axes = [ax, bx, cx, dx]
for i, season in enumerate(pd.unique(temp["season"])):

    sns.set_theme(style="whitegrid")
    df_season = temp[temp["season"] == season]
    df_season = df_season.sort_values(by=sort_value, ascending=False)
    df_season = df_season.reset_index(drop=True)
    df_season["rank"] = df_season.index + 1
    df_season["rank_perc"] = np.round(
        df_season["rank"] / len(df_season["rank"]) * 100, 1
    )

    warmest = df_season[
        np.logical_and(
            df_season["rank_perc"] >= 0, df_season["rank_perc"] <= percentage
        )
    ]


    df_freq_w = count_frequency_given_keys(warmest["bmu"], df_freq.keys())
    df_perc_w = dic_percentage(df_freq_w, len(warmest["bmu"]))

    coldest = df_season[
        np.logical_and(
            df_season["rank_perc"] >= 100 - percentage, df_season["rank_perc"] <= 100
        )
    ]

    df_freq_c = count_frequency_given_keys(coldest["bmu"], df_freq.keys())
    df_perc_c = dic_percentage(df_freq_c, len(coldest["bmu"]))


    keys = df_freq_c.keys()

    bar_width = 0.3

    r1 = np.arange(len(keys) + 1)[1:]
    r2 = [x + bar_width / 2 for x in r1]
    r0 = [x - bar_width / 2 for x in r1]
    axx = axes[i]
    axx.text(
        0.1,
        1.025,
        "(" + "abcd"[i] + ")",
        ha="center",
        va="center",
        transform=axx.transAxes,
        fontsize=20,
        fontweight="bold",
        color="k",
    )

    for i, key in enumerate(keys):
        axx.barh(
            r0[key - 1],
            df_perc_w[key],
            color=pal_hex[3],
            height=bar_width,
            edgecolor="k",
            label=str(percentage) + "% Warmest AT anomaly",
            zorder=1,
        )
        axx.barh(
            r2[key - 1],
            df_perc_c[key],
            color=pal_hex[-1],
            height=bar_width,
            edgecolor="k",
            label=str(percentage) + "% Coldest AT anomaly",
            zorder=1,
        )

    axx.set_title(season, fontsize=18, loc="center")
    axx.invert_yaxis()

    axx.set_xlim(0, 75)
    axx.set_ylabel("LSP [#]")
    axx.set_xlabel("Relative Occurence [%]")


legend_ax = fig.add_axes(
    [0.38, 0.03, 0.3, 0.05], frameon=False
)  # Adjust the position and size as needed
legend_ax.axis("off")
h, l = axx.get_legend_handles_labels()
legend_ax.legend(h[0:2], l[0:2], ncol=2, loc="center", fontsize=18, frameon=False)


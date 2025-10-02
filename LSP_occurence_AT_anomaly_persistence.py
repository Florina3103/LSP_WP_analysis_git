# %% import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# import defaultdict
from matplotlib.patches import Patch
import os
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple

# define plot parameters
params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (18, 9),
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "lines.linewidth": 4,
    "lines.markersize": 10,
    "hatch.linewidth": 0.1,
    "legend.title_fontsize": "x-large",
}
plt.rcParams.update(params)

# %% Define functions
def select_data_between_years(df, start_year, end_year):
    return df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]


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


# Function to analyze the order in the array
def analyze_order(arr, step=1):
    # Initialize a dictionary to store frequencies
    frequencies = defaultdict(lambda: defaultdict(int))

    # Iterate through the array
    for i in range(len(arr) - step):
        # Get the current and next numbers
        current_num = arr[i]
        next_num = arr[i + step]

        # Increment the frequency count for the follow-up number
        frequencies[current_num][next_num] += 1

    return frequencies


# Define a function to map months to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def welch_ttest(a, b):
    statistic = stats.ttest_ind(a, b, equal_var=False)
    alpha = 0.05

    pvalue = statistic.pvalue

    if pvalue < alpha:
        print(
            """Null hypothesis rejected -> mean AT anomaly of one node in period 1 and period 2 is significantly different"""
        )
    else:
        print(
            """Null hypothesis not rejected-> mean AT anomaly of one node in period 1 and period 2 is NOT significantly different"""
        )

    return pvalue


def consecutive_succession_distribution(df, column="bmu"):
    # Get the column values as a list
    values = df[column].tolist()

    # Find unique values in the column
    unique_values = set(values)

    # Dictionary to hold the distribution for each unique value
    result = {}

    for target in unique_values:
        consecutive_counts = []
        current_count = 0

        # Iterate through the values to count consecutive occurrences of each unique value
        for i in range(len(values) - 1):
            if values[i] == target and values[i + 1] == target:
                current_count += 1
            elif current_count > 0:
                consecutive_counts.append(
                    current_count + 1
                )  # Add 1 to include the initial occurrence
                current_count = 0

        # Handle the case where the last few values are consecutive
        if current_count > 0:
            consecutive_counts.append(current_count + 1)

        # Store the distribution (list of consecutive succession counts) for the current value
        if consecutive_counts:
            result[target] = consecutive_counts
        else:
            result[target] = []  # If no consecutive occurrences, set an empty list

    return result


def analyse_at_lsp_mean_std_dic(temp):
    at_lsp_analysis_dic = {}

    for i, season in enumerate(temp["season"].unique()):
        print(season)
        mean_at = []
        std_at = []
        mean_at_anomaly = []
        std_at_anomaly = []

        at_lsp_analysis_dic[season] = {}

        for i, node in enumerate(temp["bmu"].unique()):
            temp_df = temp[temp["season"] == season]
            temp_df = temp_df[temp_df["bmu"] == node]

            mean_at.append(temp_df["AT"].mean())
            std_at.append(temp_df["AT"].std())
            mean_at_anomaly.append(temp_df["AT_anomaly_rolling"].mean())
            std_at_anomaly.append(temp_df["AT_anomaly_rolling"].std())

        nodes = temp["bmu"].unique()

        at_lsp_analysis = (
            pd.DataFrame(
                {
                    "node": nodes,
                    "mean_at": mean_at,
                    "std_at": std_at,
                    "mean_at_anomaly": mean_at_anomaly,
                    "std_at_anomaly": std_at_anomaly,
                }
            )
            .sort_values(by="node", ascending=True)
            .reset_index(drop=True)
        )

        at_lsp_analysis_dic[season] = at_lsp_analysis

    return at_lsp_analysis_dic


def average_consecutive_succession_all(df, column):
    # Get the column values as a list
    values = df[column].tolist()

    # Find unique numbers in the column
    unique_numbers = set(values)

    result = {}

    for target in unique_numbers:
        consecutive_counts = []
        current_count = 0

        for i in range(len(values) - 1):
            if values[i] == target and values[i + 1] == target:
                current_count += 1
            elif current_count > 0:
                consecutive_counts.append(current_count + 1)
                current_count = 0

        if current_count > 0:
            consecutive_counts.append(current_count + 1)

        if consecutive_counts:
            average_succession = sum(consecutive_counts) / len(consecutive_counts)
        else:
            average_succession = 0

        result[target] = average_succession  # round(average_succession)

    return result


def calculate_consecutive_bmu(
    df, method="median", bmu_column="bmu", date_column="date"
):
    
    # Datumsspalte in DateTime umwandeln
    df[date_column] = pd.to_datetime(df[date_column])

    # Ergebnis-DataFrame initialisieren
    bmu_values = sorted(df[bmu_column].unique())  # BMU-Werte sortiert
    bmu_succession_lengths = {
        bmu: [] for bmu in bmu_values
    }  # Dictionary für Abfolgenlängen

    # Für alle BMU-Werte die Abfolgen berechnen
    for bmu in bmu_values:
        bmu_data = df[df[bmu_column] == bmu].sort_values(by=date_column)
        current_streak = 1

        # Über alle Tage in den BMU-Daten iterieren
        for i in range(1, len(bmu_data)):
            # BMU des aktuellen und vorherigen Tages vergleichen
            if (
                bmu_data.iloc[i][date_column] - bmu_data.iloc[i - 1][date_column]
            ).days == 1:
                current_streak += 1  # Fortsetzen der Abfolge
            else:
                # Abfolge beenden und speichern
                bmu_succession_lengths[bmu].append(current_streak)
                current_streak = 1  # Streak zurücksetzen

        # Letzte Abfolge speichern
        bmu_succession_lengths[bmu].append(current_streak)

        # Median oder Maximalwert der Abfolgen für jeden BMU berechnen
    result = {}
    spread = {}
    for bmu, lengths in bmu_succession_lengths.items():
        if lengths:
            if method == "median":
                result[bmu] = np.median(lengths)
                spread[bmu] = np.std(lengths)
            elif method == "max":
                result[bmu] = np.max(lengths)
                spread[bmu] = np.std(lengths)
            elif method == "mean":
                result[bmu] = np.mean(lengths)
                spread[bmu] = np.std(lengths)
        else:
            result[bmu] = np.nan
            spread[bmu] = np.nan

    # Rückgabe als DataFrame
    result_df = pd.DataFrame(
        {
            bmu_column: result.keys(),
            f"{method}_succession_length": result.values(),
            f"{method}_spread": spread.values(),
        }
    )

    result_df.set_index(bmu_column, inplace=True)

    return result_df, bmu_succession_lengths


# %% read in data
station = "WEG_L"
pal = sns.color_palette("colorblind")
pal_hex = pal.as_hex()

path = os.getcwd()

# Read AT at station from 20CRv3
df1 = pd.read_csv(path + f".../AT_20CRv3_{station}_daily.csv")

# Read the SOM data
df2 = pd.read_csv(
    path + "/.../bmu_SOM_8_ssim_hgt_GRl_1900_2015.csv"
)

# Convert the 'date' column to datetime format
df1["date"] = pd.to_datetime(df1["time"])
df2["date"] = pd.to_datetime(df2["time"])

# Merge the two DataFrames on the 'date' column
merged_df = pd.merge(df1, df2, on="date", how="inner")
merged_df = merged_df[["date", "AT", "bmu"]]

# %% Prepare the data
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

method = 'mean'

fp_consecutive_days, fp_consecutive_days_distribution = calculate_consecutive_bmu(merged_df, method = method, bmu_column='bmu', date_column='date')
wp1_consecutive_days, wp1_consecutive_days_distribution = calculate_consecutive_bmu(df1, method = method, bmu_column='bmu', date_column='date')
wp2_consecutive_days, wp2_consecutive_days_distribution = calculate_consecutive_bmu(df2, method = method, bmu_column='bmu', date_column='date')

consecutive_per_bmu = consecutive_succession_distribution(merged_df)
consecutive_per_bmu_wp1 = consecutive_succession_distribution(df1)
consecutive_per_bmu_wp2 = consecutive_succession_distribution(df2)
# ignores the instances where the LSP is not consecutive so therefore use calculate_consecutive_bmu function

data = []
for label, bmu_data in zip(
    ["consecutive_bmu", "consecutive_bmu_wp1", "consecutive_bmu_wp2"],
    [
        fp_consecutive_days_distribution,
        wp1_consecutive_days_distribution,
        wp2_consecutive_days_distribution,
    ],
):
    #    [consecutive_per_bmu, consecutive_per_bmu_wp1, consecutive_per_bmu_wp2]):
    for bmu, values in bmu_data.items():
        data.extend([[bmu, v, label] for v in values])
consecutive_df = pd.DataFrame(data, columns=["BMU", "Value", "Source"])

# %% PLOT FIG 4

summary = merged_df.groupby(["Period", "bmu"]).agg(
    {"bmu": ["sum"], "AT_anomaly_rolling": ["mean", "std"]}
)
merged_df["year"] = merged_df["date"].dt.year

# Group by year and bmu (the cluster) and compute the average AT_anomaly_rolling for each cluster-year combination
grouped_df = (
    merged_df.groupby(["year", "bmu"]).agg({"AT_anomaly_rolling": "mean"}).reset_index()
)

# Pivot the data so we can plot it easily as a heatmap (years on x-axis, clusters on y-axis)
heatmap_data = grouped_df.pivot(
    index="bmu", columns="year", values="AT_anomaly_rolling"
)

# Example settings
size = 100
bar_width = 0.25
marker = "s"

# Prepare positions for the different periods
r0 = np.arange(1, len(np.unique(merged_df["bmu"])) + 1)[0:]
r1 = [x - bar_width for x in r0]
r2 = [x + bar_width for x in r0]

x_values = [r1[::-1], r0[::-1], r2[::-1]]
marker = ["s", "o", "^"]
labels = ["1900-2015", "WP1 1922-1932", "WP2 1993-2007"]

fig = plt.figure(figsize=(15, 15), constrained_layout=True)
gs = GridSpec(4, 3, height_ratios=[8, 0.05, 6, 0.2], width_ratios=[1, 1, 1], figure=fig)

# Subplot 1: Horizontal Bar Plot (on the left)
ax = fig.add_subplot(gs[0, 0])
cx = fig.add_subplot(gs[0, 1])
dx = fig.add_subplot(gs[0, 2])
bx = fig.add_subplot(gs[2, 0:3])

keys = df_freq.keys()

# Define the width of bars and positions
bar_width = 0.3
r1 = np.arange(len(keys) + 1)[1:]  # Define y positions
r2 = [x + bar_width for x in r1]
r0 = [x - bar_width for x in r1]
ax.grid(True, zorder=-1)
# Plot horizontal bars, but negate the values so they extend from right to left
for i, key in enumerate(sorted(keys)):
    ax.barh(
        r0[key - 1],
        df_perc[key],
        color=pal_hex[0],
        height=bar_width,
        edgecolor="k",
        label="1900-2015",
        zorder=2,
    )
    ax.barh(
        r1[key - 1],
        df1_perc[key],
        color=pal_hex[1],
        height=bar_width,
        edgecolor="k",
        label="WP1 1922-1932",
        zorder=2,
    )
    ax.barh(
        r2[key - 1],
        df2_perc[key],
        color=pal_hex[2],
        height=bar_width,
        edgecolor="k",
        label="WP2 1993-2007",
        zorder=2,
    )

# Adjust the y-axis and invert it
ax.set_ylim(0.5, 8.5)
ax.set_yticks(r1)

ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_xlabel("Relative Occurrence [%]")


# Invert the x-axis so bars go from right to left
ax.invert_yaxis()  # Invert y-axis as well to match ordering
ax.invert_xaxis()  # Invert x-axis to have 0% on the right
ax.set_yticklabels(r1, va="center")

# Subplot 2: Heatmap (in the middle)
# Heatmap

# Plot heatmap with the colorbar beneath
cbar_ax = fig.add_axes(
    [0.1, -0.02, 0.8, 0.02]
)  # Adjust the position and size as needed
sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    vmin=-10,
    vmax=10,
    cbar_kws={
        "label": "AT Anomaly [°C]",
        "extend": "both",
        "orientation": "horizontal",
    },
    ax=bx,
    cbar_ax=cbar_ax,
)  # Place the colorbar below the heatmap
cbar_ax.spines["outline"].set(visible=True, lw=1, edgecolor="black")
# Adjust the x-axis to show ticks every 5 years
bx.set_xticks(np.arange(0, len(heatmap_data.columns), 5) + 0.5)
bx.set_xticklabels(heatmap_data.columns[::5], rotation=45, ha="center")

bx.set_yticklabels(
    bx.get_yticklabels(), rotation="horizontal", rotation_mode="anchor", va="center"
)  # Set labels for heatmap

# Set labels for heatmap
bx.set_xlabel("Year")
bx.set_ylabel("LSP [#]")


bx.vlines(
    np.where(heatmap_data.columns == 1900)[0][0], *bx.get_ylim(), color=pal_hex[0], lw=8
)  # WP2 Start
bx.vlines(116, *bx.get_ylim(), color=pal_hex[0], lw=8)  # WP2 End
bx.hlines(
    8, *bx.get_xlim(), color=pal_hex[0], lw=8
)  # Add horizontal lines to separate the clusters
bx.hlines(0, *bx.get_xlim(), color=pal_hex[0], lw=8)

bx.vlines(
    np.where(heatmap_data.columns == 1922)[0][0], *bx.get_ylim(), color=pal_hex[1], lw=6
)  # WP1 Start
bx.vlines(
    np.where(heatmap_data.columns == 1933)[0][0], *bx.get_ylim(), color=pal_hex[1], lw=6
)  # WP1 End
bx.vlines(
    np.where(heatmap_data.columns == 1993)[0][0], *bx.get_ylim(), color=pal_hex[2], lw=6
)  # WP2 Start
bx.vlines(
    np.where(heatmap_data.columns == 2008)[0][0], *bx.get_ylim(), color=pal_hex[2], lw=6
)  # WP2 End

bx.hlines(
    8,
    (np.where(heatmap_data.columns == 1922)[0][0]),
    (np.where(heatmap_data.columns == 1933)[0][0]),
    color=pal_hex[1],
    lw=8,
)  # Add horizontal lines to separate the clusters
bx.hlines(
    0,
    (np.where(heatmap_data.columns == 1922)[0][0]),
    (np.where(heatmap_data.columns == 1933)[0][0]),
    color=pal_hex[1],
    lw=8,
)
bx.hlines(
    8,
    (np.where(heatmap_data.columns == 1993)[0][0]),
    (np.where(heatmap_data.columns == 2008)[0][0]),
    color=pal_hex[2],
    lw=8,
)  # Add horizontal lines to separate the clusters
bx.hlines(
    0,
    (np.where(heatmap_data.columns == 1993)[0][0]),
    (np.where(heatmap_data.columns == 2008)[0][0]),
    color=pal_hex[2],
    lw=8,
)


# Subplot 3: Errorbar Plot (on the right)
# Errorbar plot on the right
cx.grid(True, zorder=-1)
# Vertical line at AT Anomaly = 0
cx.set_yticklabels(r1, va="center")
# Plot the error bars for the different periods
for i, period in enumerate(summary.index.get_level_values("Period").unique()):
    cx.errorbar(
        summary.loc[period]["AT_anomaly_rolling"]["mean"][::-1],  # x-values
        x_values[i],  # y-values (LSPs)
        xerr=summary.loc[period]["AT_anomaly_rolling"]["std"][
            ::-1
        ],  # Horizontal error bars (xerr)
        fmt="o",
        color=pal_hex[i],
        label=period,
        zorder=3,
        linewidth=3,
        marker=marker[i],
        markeredgecolor="k",
        capsize=7,
        capthick=2,
        barsabove=True,
        dash_capstyle="butt",
    )
    cx.scatter(
        summary.loc[period]["AT_anomaly_rolling"]["mean"][
            ::-1
        ],  # x-values (Mean AT Anomaly)
        x_values[i],  # y-values (LSPs)
        s=size,
        edgecolor="k",
        color=pal_hex[i],
        marker=marker[i],
        zorder=4,
    )
cx.axvline(x=0, c="k", lw=2, zorder=2)
# Adjust the y-axis to be on the right side and inverted
# ax.yaxis.set_label_position("right")
# ax.yaxis.tick_right()
cx.set_ylim(0.5, 8.5)
cx.set_yticks(r1)
cx.invert_yaxis()
cx.set_ylabel("LSP [#]")
cx.set_xlabel("Mean AT Anomaly [°C]")


cx2 = cx.twinx()
cx2.set_ylim(cx.get_ylim())
cx2.set_yticks(cx.get_yticks())
cx2.set_yticklabels(cx.get_yticklabels(), va="center")

# Subplot 4: Violin Plot (on the far right)

dx.grid(True, zorder=-1)

dx.invert_yaxis()

# Violinplot with seaborn, this time with BMU on the y-axis and Value on the x-axis
side_con = ["low", "high", "high"]
alphas = [0.6, 0.6, 0.4]
for s, source in enumerate(np.unique(consecutive_df["Source"])):
    for b, bmu in enumerate(np.unique(consecutive_df["BMU"])):
        data = consecutive_df[
            (consecutive_df["BMU"] == bmu) & (consecutive_df["Source"] == source)
        ]["Value"]
        violin_parts = dx.violinplot(
            data,
            positions=[r1[b]],
            widths=0.75,
            side=side_con[s],
            showmeans=True,
            showextrema=False,
            showmedians=False,
            vert=False,
        )  # , color = pal_hex[s])

        for pc in violin_parts["bodies"]:
            pc.set_facecolor(pal_hex[s])
            pc.set_edgecolor(pal_hex[s])
            pc.set_alpha(alphas[s])
            pc.set_zorder(2)
        violin_parts["cmeans"].set_colors(pal_hex[s])
        lines = violin_parts["cmeans"].get_segments()
        new_lines = []
        for line in lines:
            if s > 0:
                min_line = line.min(axis=0)
                line = (line - min_line) * np.array([1, 1]) + min_line + 0.028
            else:
                max_line = line.max(axis=0) - 0.028
                line = (line - max_line) * np.array([1, 1]) + max_line - 0.028
            new_lines.append(line)
        violin_parts["cmeans"].set_segments(new_lines)
        violin_parts["cmeans"].set_zorder(3)

# Adjust axes and title
dx.set_ylabel("LSP [#]")
dx.set_xlabel("Persistence [days]")
dx.set_yticks(r1)

dx.legend().remove()
dx.set_xlim(1, 10)
dx.set_ylim(8.5, 0.5)
dx.set_yticklabels(r1, va="center")
dx.set_xticks([2, 4, 6, 8])

legend_periods_add = [
    Patch(facecolor=pal_hex[0], edgecolor="k", label="1900-2015"),
    Patch(facecolor=pal_hex[1], edgecolor="k", label="WP1 1922-1932"),
    Patch(facecolor=pal_hex[2], edgecolor="k", label="WP2 1993-2007"),
]
# Legend below second plot
h, l = cx.get_legend_handles_labels()
# legend_ax = fig.add_axes([0.17, 0.52, 0.5, 0.04])
# bx.legend(h[0:3], l[0:3], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
bx.legend(
    [
        (legend_periods_add[0], h[0]),
        (legend_periods_add[1], h[1]),
        (legend_periods_add[2], h[2]),
    ],
    (l[0], l[1], l[2]),
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3,
    handletextpad=2,
    frameon=False,
    columnspacing=3,
    handler_map={tuple: HandlerTuple(ndivide=1.5)},
    fontsize=20,
)


ax.text(
    0.1,
    0.96,
    "(a)",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=20,
    fontweight="bold",
    color="k",
)
bx.text(
    0.035,
    0.93,
    "(d)",
    ha="center",
    va="center",
    transform=bx.transAxes,
    fontsize=20,
    fontweight="bold",
    color="k",
)
cx.text(
    0.1,
    0.96,
    "(b)",
    ha="center",
    va="center",
    transform=cx.transAxes,
    fontsize=20,
    fontweight="bold",
    color="k",
)
dx.text(
    0.1,
    0.96,
    "(c)",
    ha="center",
    va="center",
    transform=dx.transAxes,
    fontsize=20,
    fontweight="bold",
    color="k",
)


plt.show()

# %% PLOT FIG 5
merged_df["year"] = merged_df["date"].dt.year
# Define the periods to exclude
exclude_periods = [(1922, 1932), (1997, 2007)]

# Filter out rows where the 'year' falls within the specified periods
for start, end in exclude_periods:
    df_ex_wp = merged_df[(merged_df["year"] < start) | (merged_df["year"] > end)]

fig = plt.figure(figsize=(15, 8), constrained_layout=True)
gs = GridSpec(2, 4, height_ratios=[7, 1], width_ratios=[1, 1, 1, 1], figure=fig)

# Subplot 1: Horizontal Bar Plot (on the left)
ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[0, 1])
cx = fig.add_subplot(gs[0, 2])
dx = fig.add_subplot(gs[0, 3])

axes = [ax, bx, cx, dx]

month_season = [" DJF", " MAM", " JJA", " SON"]
save = False
periods = [df1, df2]

for i, season in enumerate(pd.unique(merged_df["season"])):

    FP = merged_df[merged_df["season"] == season]
    FP_freq = count_frequency(FP["bmu"])
    FP_freq = dict(sorted(FP_freq.items()))

    keys = FP_freq.keys()

    bar_width = 0.3

    r1 = np.arange(len(keys) + 1)[1:]
    r2 = [x + bar_width / 2 for x in r1]
    r0 = [x - bar_width / 2 for x in r1]

    axx = axes[i]
    axx.grid(True, zorder=0)
    x_values = [r0, r2]
    axx.axvline(1, color="k", lw=1, zorder=2)

    for j, period in enumerate(periods):
        temp = period[period["season"] == season]
        temp_freq = count_frequency(temp["bmu"])
        temp_freq = dict(sorted(temp_freq.items()))
        for key in temp_freq.keys():
            temp_freq[key] = (temp_freq[key] / len(temp)) / (FP_freq[key] / len(FP))
            axx.barh(
                x_values[j][key - 1],
                temp_freq[key],
                color=pal_hex[j + 1],
                height=bar_width,
                edgecolor="k",
                zorder=2,
                label=temp["Period"].iloc[0],
            )
    axx.set_title(season + month_season[i], fontsize=18, loc="center")
    axx.set_ylabel("LSP [#]")
    axx.set_ylim(0.5, 8.5)
    axx.invert_yaxis()
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

legend_ax = fig.add_axes(
    [0.38, 0.03, 0.3, 0.05], frameon=False
)  
legend_ax.axis("off")

h, l = ax.get_legend_handles_labels()
legend_ax.legend(
    [h[0], h[-1]],
    [l[0], l[-1]],
    loc="center",
    labelspacing=1,
    frameon=False,
    ncol=2,
    fontsize=18,
    title="Relative Occurrence compared to 1900-2015",
    title_fontsize=18,
)

# %%

# %%
import datetime
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from scipy.stats import theilslopes

sys.path.append("..")
import cartopy.crs as ccrs
import matplotlib.path as mpath
import pymannkendall as mk

import xarray_clim
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# %% Define functions


def get_timespan(year1, year2):
    return list(range(year1, year2 + 1))


def correct_timeformat(ds):
    origin = pd.DatetimeIndex(np.array([ds.time.units[13:]]))[0]

    time = []
    for t in ds.time.values:
        time.append(
            datetime.date(origin.year, origin.month, origin.day)
            + relativedelta(months=int(t))
        )

    ds["time"] = pd.DatetimeIndex(time)

    return ds

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def open_files(year1, year2, path):
    """
    year1,year2 as int
    path as str
    variables as ['str','str',....]
    """

    data = {}
    years = get_timespan(year1, year2)
    for i, year in enumerate(years):
        ds = xr.open_dataset(
            path + str(year) + ".nc", decode_times=False
        )  # vars = variables,
        ds = correct_timeformat(ds)
        ds = ds.resample(time="Y").mean("time")
        data[year] = ds
        print("MAR data from year " + str(year) + " is loaded.")

    data = xr.concat(list(data.values()), dim="time")

    return data


def open_20CR(year1, year2, path):
    """
    year1,year2 as int
    path as str
    variables as ['str','str',....]
    """

    data = {}
    years = get_timespan(year1, year2)

    for i, year in enumerate(years):
        ds = xr.open_dataset(path + "air.2m." + str(year) + ".nc")
        ds = xarray_clim.wrap360_to180(ds)
        ds = ds.resample(time="Y").mean("time")
        ds["lat"] = ds["lat"].astype(dtype="float64")
        ds["lon"] = ds["lon"].astype(dtype="float64")
        gl = xarray_clim.sellonlatbox(ds, -100, 90, 0, 55)
        data[year] = gl
        print("20CRv3 data from year " + str(year) + " is loaded.")

    data = xr.concat(list(data.values()), dim="time")

    return data


def data_processing(ds):
    # Access the data variable for which you want to calculate linear regression
    data_variable = ds["air"]  #'air'

    slopes = np.zeros_like(data_variable.values[0])
    trend_significance = np.zeros_like(data_variable.values[0])

    # Calculate linear regression at each grid point
    for i in range(data_variable.shape[1]):
        for j in range(data_variable.shape[2]):
            y = data_variable[:, i, j].values
            x = np.arange(len(y))

            valid_mask = ~np.isnan(y)

            if np.sum(valid_mask) > 1:
                # result = linregress(x[valid_mask], y[valid_mask])
                result = theilslopes(y[valid_mask], x[valid_mask], 0.9)
                slopes[i, j] = result.slope

                significance_value = mk.original_test(y)[2]
                trend_significance[i, j] = significance_value

    threshold = 0.05
    significant_yes_no = np.where(trend_significance < threshold, 1, 0)

    return slopes, significant_yes_no

# %%
# Setting plotting parameters globally for consistency
params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.title_fontsize": "x-large",
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
    "hatch.linewidth": 0.1,
}
plt.rcParams.update(params)

# %% define station data
station_name = "WEG_L"
path = os.getcwd()


# File paths and input details
path = "/nas/data/raw/dmi/historical_climate_data_collection_Greenland_updated_until_2020/Monthly/"
csv_file = "gr_monthly_all_1784_2020.csv"

stations = [4211, 4221, 4250, 4272, 4360]
names = ["UPV", "ILU", "NUK", "QAQ", "TAS"]
variables = [101]
years = [1900, 2015]


# %% read in 20CRv3 data and process

path_20CR = "/nas/data/raw/noaa/20Cv3/air.2m/"

# smoothed data
period1_smoothed = open_20CR(1917, 1937, path_20CR)
period2_smoothed = open_20CR(1988, 2012, path_20CR)

#%% 
#data processing of the 20CRv3 data


period1_smoothed = period1_smoothed.rolling(time=5).mean()
period2_smoothed = period2_smoothed.rolling(time=5).mean()

period1_smoothed = period1_smoothed.sel(time=slice('1922-12-31T00:00:00.000000000','1932-12-31T00:00:00.000000000'))
period2_smoothed = period2_smoothed.sel(time=slice('1993-12-31T00:00:00.000000000','2007-12-31T00:00:00.000000000'))

slope1_smoothed, sig1_smoothed = data_processing(period1_smoothed)
slope2_smoothed, sig2_smoothed = data_processing(period2_smoothed)

lon_2d, lat_2d = np.meshgrid(period1_smoothed.lon, period1_smoothed.lat)

# %% read in and process station data
station_name = "WEG_L"
path = '/home/flo/LSP_analysis'#os.getcwd()
df_weg = pd.read_csv(path + f"/Data/AT_20CRv3_{station_name}_daily.csv")
df_weg["date"] = pd.to_datetime(df_weg["time"])
df_weg["year"] = df_weg["date"].dt.year
df_weg["month"] = df_weg["date"].dt.month
df_weg["season"] = df_weg['month'].map(get_season)
df_weg_year = df_weg.groupby("year")["AT"].mean().reset_index()
at_mean = np.nanmean(df_weg_year['AT'][-30:])
at_ano_re_weg = df_weg_year - at_mean
df_weg_year['AT_ano'] = df_weg_year['AT'] - at_mean
df_weg_year['AT_ano_smooth'] = df_weg_year["AT_ano"].rolling(window=5, min_periods=1).mean()
at_ano_re_weg = at_ano_re_weg.rolling(window=5, min_periods=1).mean()

path = "/nas/data/raw/dmi/historical_climate_data_collection_Greenland_updated_until_2020/Monthly/"
csv_file = "gr_monthly_all_1784_2020.csv"

df = pd.read_csv(path + csv_file, delimiter=";")
df.iloc[:, 3:16] = df.iloc[:, 3:16].applymap(lambda x: float(str(x).replace(",", ".")))


window = 5  # Window size for smoothing
year_window = 5  # Window size for calculating temperature difference
aws = {"at_ref": {}, "at_ano": {}, "at": {}, f"at_diff_{year_window}_years": {}}

at_diff_re_weg = at_ano_re_weg.diff(year_window) / year_window

# Loop through each station, calculate anomaly and smoothed values
for i, station in enumerate(stations):
    temp_data = df[
        (df["stat_no"] == station)
        & (df["elem_no"].isin(variables))
        & (df["year"].between(years[0], years[1]))
    ].reset_index(drop=True)
    aws[names[i]] = temp_data

    # Calculate reference temperature (last 30 years)
    at_mean = np.nanmean(temp_data["annual"][-30:])
    aws["at_ref"][names[i]] = at_mean

    # Calculate anomaly and apply rolling mean for smoothing
    aws["at_ano"][names[i]] = temp_data["annual"] - at_mean
    aws[names[i]]['at_ano'] = aws["at_ano"][names[i]]
    aws["at"][names[i]] = (
        temp_data["annual"].rolling(window=window, min_periods=1).mean()
    )
    aws[names[i]]["annual_smooth"] = (
        aws["at_ano"][names[i]].rolling(window=window, min_periods=1).mean()
    )
    aws[f"at_diff_{year_window}_years"][names[i]] = (
        temp_data["annual"].diff(year_window) / year_window
    )


# Load zonal mean data and process anomalies
zonal_mean = pd.read_csv(
    "/home/flo/LSP_analysis/Data/20CRv3_zonal_mean_yearly.csv", delimiter=","
)
for col in zonal_mean.columns[1:]:
    zonal_mean[col] = zonal_mean[col] - 273.15
    zonal_mean[col + "_ano"] = zonal_mean[col] - np.nanmean(zonal_mean[col][-30:])
    zonal_mean[col + "_at"] = (
        zonal_mean[col].rolling(window=window, min_periods=1).mean()
    )
    zonal_mean[col + "_smooth"] = (
        zonal_mean[col + "_ano"].rolling(window=window, min_periods=1).mean()
    )

smooth_columns = [col for col in zonal_mean.columns if col.endswith("_smooth")]
smooth_columns.pop(1)
at_columns = [col for col in zonal_mean.columns if col.endswith("_at")]

# %% Define map section 

myProj = myProj = ccrs.NorthPolarStereo(central_longitude=-50)
myProj._threshold = myProj._threshold / 40.0  # for higher precision plot
noProj = ccrs.PlateCarree(central_longitude=0)


fig = plt.figure(figsize=(8, 12))
ax = fig.add_subplot(1, 1, 1, projection=myProj)

[ax_hdl] = ax.plot(
    [-100, -75, -50, -25, -1, -1, -1, -1, -25, -50, -75, -100, -100, -100],
    [55, 55, 55, 55, 55, 55, 90, 90, 90, 90, 90, 90, 90, 55],
    color="black",
    linewidth=0.5,
    transform=noProj,
)
tx_path = ax_hdl._get_transformed_path()
path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
polygon1s = mpath.Path(path_in_data_coords.vertices)


# %% plot FIG 2


colors_hex = ["#443983", "#31688e", "#21918c", "#35b779", "#90d743"]
line_styles = [":", "--", "-.", "-"]
label_rean = ["Global", "Arctic", "Greenland"]

selected_smooth_columns = smooth_columns  # Add your actual smooth column data

# Create the combined figure with gridspec
fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(2, 7, height_ratios=[2, 2], width_ratios=[1, 1, 1, 1, 1, 1, 1])
gs.update(wspace=0.3, hspace=0.3)

# Top panel (full-width)
ax_top = fig.add_subplot(gs[0, :])
ax_top.grid(True, zorder=1)
ax_top.axhline(0, color="black", zorder=2, lw=1)

# Plot shaded regions for specific periods
ax_top.fill_betweenx((-3, 3), 1922, 1932, color="grey", alpha=0.5)
ax_top.fill_betweenx((-3, 3), 1993, 2007, color="grey", alpha=0.5)

# Plot station data with smoothing
for i, station in enumerate(names):
    ax_top.plot(
        aws[station]["year"],
        aws[station]["annual_smooth"],
        color=colors_hex[i],
        zorder=3,
        label=station,
        lw=2.5,
    )

# Plot zonal mean smoothed values
for j, smooth_col in enumerate(selected_smooth_columns):
    ax_top.plot(
        zonal_mean["years"],
        zonal_mean[smooth_col],
        color="#8e2d04",
        label=label_rean[j],
        zorder=3,
        linestyle=line_styles[j],
        lw=2.5,
    )
ax_top.plot(
    df_weg_year["year"],
    df_weg_year["AT_ano_smooth"],
    color="#8e2d04",
    lw=2.5,
    zorder=3,
    linestyle=(0, (3, 1, 1, 1, 1, 1)),
    label=station_name,
)


ax_top.text(
    -0.05,
    0.96,
    "(a)",
    ha="center",
    va="center",
    transform=ax_top.transAxes,
    fontsize=20,
    fontweight="bold",
    color="k",
)
# Customizing the legend with better positioning
h, l = ax_top.get_legend_handles_labels()
ax_top.legend(
    h[:5],
    l[:5],
    bbox_to_anchor=(0.45, -0.1),
    title="Observations",
    ncol=3,
    frameon=False,
    title_fontsize=18,
    fontsize=18,
)

# Adding a secondary y-axis for reanalysis data legend
ax2 = ax_top.twinx()
ax2.get_yaxis().set_visible(False)
ax2.legend(
    h[5:],
    l[5:],
    bbox_to_anchor=(0.95, -0.1),
    handlelength=3,
    title="Reanalysis",
    ncol=2,
    frameon=False,
    title_fontsize=18,
    fontsize=18,
)

# Setting axis labels and limits
ax_top.set_xlabel("Year")
ax_top.set_ylabel("AT anomaly [°C]")
ax_top.set_xlim(1900, 2010)
ax_top.set_ylim(-2.6, 1.8)
ax_top.set_yticks(np.arange(-2, 2, 1))

ax_top.text(1923, 1.5, "WP1", fontweight="bold", fontsize=18)
ax_top.text(1994, 1.5, "WP2", fontweight="bold", fontsize=18)

# Bottom panels
myProj._threshold /= 40.0  # For higher precision plot

# Set color normalization centered around 0
norm = mpl.colors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.3)

my_list = np.arange(-0.2, 0.3, 0.1).tolist()
cb1 = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)


# First map panel
ax1 = fig.add_subplot(gs[1, 0:3], projection=myProj)
ax1.contourf(
    period1_smoothed.lon,
    period1_smoothed.lat,
    slope1_smoothed,
    transform=ccrs.PlateCarree(),
    cmap="coolwarm",
    norm=norm,
)  # norm=colors.CenteredNorm(0))plt.cm.RdBu_r
ax1.coastlines()
ax1.set_boundary(polygon1s)  # Masks out unwanted part of the plot
gl = ax1.gridlines(draw_labels=False)

ax1.text(-84, 48.6, "80°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=-30)
ax1.text(-64, 51, "60°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=-8)
ax1.text(-44, 52, "40°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=15)
ax1.text(-24, 53, "20°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=30)

ax1.text(-20, 80, "80°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)
ax1.text(-10, 70, "70°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)
ax1.text(-7, 60, "60°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)

ax1.set_title("1922-1932 WP1", fontweight="bold")
ax1.text(
    -0.02,
    1.06,
    "(b)",
    ha="center",
    va="center",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold",
    color="k",
)
ax1.scatter(
    -51.128333,
    71.140556,
    edgecolor="k",
    s=150,
    c="green",
    zorder=14,
    transform=ccrs.PlateCarree(),
    label="study site",
) 


cs = ax1.contourf(
    period1_smoothed.lon,
    period1_smoothed.lat,
    sig1_smoothed,
    1,
    colors="none",
    transform=ccrs.PlateCarree(),
    hatches=[None, ".."],
)

bx = fig.add_subplot(gs[1, 3:6], projection=myProj)
bx.contourf(
    period2_smoothed.lon,
    period2_smoothed.lat,
    slope2_smoothed,
    transform=ccrs.PlateCarree(),
    cmap="coolwarm",
    norm=norm,
)  
bx.set_boundary(polygon1s)  # masks-out unwanted part of the plot
bx.coastlines()

gl = bx.gridlines(draw_labels=False)

bx.text(-84, 48.6, "80°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=-30)
bx.text(-64, 51, "60°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=-8)
bx.text(-44, 52, "40°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=15)
bx.text(-24, 53, "20°W", transform=ccrs.PlateCarree(), fontsize=18, rotation=30)

bx.text(-20, 80, "80°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)
bx.text(-10, 70, "70°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)
bx.text(-7, 60, "60°N", transform=ccrs.PlateCarree(), fontsize=18, rotation=-40)

bx.set_title("1993-2007 WP2", fontweight="bold")
bx.text(
    -0.02,
    1.06,
    "(c)",
    ha="center",
    va="center",
    transform=bx.transAxes,
    fontsize=20,
    fontweight="bold",
    color="k",
)

cs = bx.contourf(
    period1_smoothed.lon,
    period1_smoothed.lat,
    sig2_smoothed,
    1,
    colors="none",
    transform=ccrs.PlateCarree(),
    hatches=[None, ".."],
)

bx.scatter(
    -51.128333,
    71.140556,
    edgecolor="k",
    s=150,
    c="green",
    zorder=14,
    transform=ccrs.PlateCarree(),
    label="study site",
)  # add right coordinates



bx.legend(
    bbox_to_anchor=(1.44, 0.1),
    handlelength=0.5,
    handletextpad=1.5,
    labelspacing=0.08,
    markerscale=1,
    fontsize=18,
    frameon=False,
)

legend_elements = [
    mpatches.Patch(
        facecolor="none", edgecolor="black", hatch="..", label="significant"
    ),  # Custom hatch legend
]

# Add the legend
ax1.legend(
    bbox_to_anchor=(2.535, -0.02), handles=legend_elements, frameon=False, fontsize=18
)

cax = fig.add_axes([0.8, 0.2, 0.015, 0.2])
cbar = fig.colorbar(cb1, cax=cax, extend="both", norm=norm)
cbar.set_label("AT trend [°C/year]", rotation=90, labelpad=15)

cbar.ax.set_yticklabels(np.round(np.arange(-0.2, 0.3, 0.1), 1), fontsize=18)
for t in cbar.ax.get_yticklabels():
    t.set_horizontalalignment("right")
    t.set_x(3.5)

cbar.ax.set_yscale("linear")


plt.tight_layout(rect=[0, 0, 0.5, 1])


plt.show()



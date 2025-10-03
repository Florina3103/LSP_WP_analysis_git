# %%

# from scipy.signal import savgol_filter
import glob

import numpy as np
import pandas as pd
import xarray as xr

import xarray_clim

# %%

# load 20CRv3 data to find zonal mean AT anomaly

path = "/nas/data/raw/noaa/20Cv3/air.2m/"

# Use glob to get a list of all NetCDF files in the folder
nc_files = glob.glob(f"{path}/air.2m.*.nc")
nc_files.sort()

year1 = 1900
year2 = 2015

valid_nc_files = []
for file in nc_files:
    file_year = int(file[-7:-3])
    if year1 <= file_year <= year2:
        valid_nc_files.append(file)
# %%
# loop over each year to get average temperature per year for the globe, NH and Arctic
global_AT = []
arctic_AT = []
gl_AT = []
years = []
time = []

for i, file in enumerate(valid_nc_files):
    # load the data per year

    ds = xr.open_dataset(file)
    ds = xarray_clim.wrap360_to180(ds)
    ds = ds.resample(time="Y").mean("time")
    ds["lat"] = ds["lat"].astype(dtype="float64")
    ds["lon"] = ds["lon"].astype(dtype="float64")
    print("file " + file[-7:-3] + " is loaded")

    # select the regions of interest:

    arctic = xarray_clim.sellonlatbox(ds, -180, 90, 179, 66.5)
    gl = xarray_clim.sellonlatbox(ds, -75, 85, -6, 58)
    print("regions are selected")

    # calculate the zonal mean AT of each region

    at_g = (ds["air"].weighted(np.cos(np.deg2rad(ds.lat)))).mean(dim=["lon", "lat"])
    at_arctic = (arctic["air"].weighted(np.cos(np.deg2rad(arctic.lat)))).mean(
        dim=["lon", "lat"]
    )
    at_gl = (gl["air"].weighted(np.cos(np.deg2rad(gl.lat)))).mean(dim=["lon", "lat"])
    print("yearly zonal mean is calculated")

    global_AT.extend(at_g.values)
    arctic_AT.extend(at_arctic.values)
    gl_AT.extend(at_gl.values)
    time.extend(at_g.time.values)


year = list(range(year1, year2 + 1))
df_zonal_means = pd.DataFrame(
    {"year": year, "global_AT": global_AT, "arctic_AT": arctic_AT, "gl_AT": gl_AT}
)
df_zonal_means.to_csv("20CRv3_zonal_mean_annual.csv", index=False)

# %%

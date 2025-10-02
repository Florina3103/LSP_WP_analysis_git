#%%
import pandas as pd
import xarray as xr
import numpy as np

import xarray_clim

#%%


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


def open_20CR(year1, year2, path, limits = [-100, 90, 0, 55]):
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
        gl = xarray_clim.sellonlatbox(ds, *limits)
        data[year] = gl
        print("20CRv3 data from year " + str(year) + " is loaded.")

    data = xr.concat(list(data.values()), dim="time")

    return data

# %%


path_20CR = "/nas/data/raw/noaa/20Cv3/air.2m/"

limit_Greenland = [-75, 85, -6, 58]
limit_Arctic = [-180, 180, 66.5, 90]

year1 = 1900
year2 = 1901

# smoothed data
data_20CRv3 = open_20CR(year1, year2, path_20CR, limit_Greenland)

for time in data_20CRv3['time']:
    month = pd.to_datetime(str(time.values)).month
    season = get_season(month)
    data_20CRv3['season'] = season
zonal_average = data_20CRv3['air'].mean(weighted=True, dim='lon')

# %%

# %%
import pandas as pd
import xarray as xr
import numpy as np


# %%

reanalysis_folder = ".../noaa/20Cv3"  

station_coords = {
    "WEG_L": {"lat": 71.2, "lon": -51.133333, "alt": 940},
    "WEG_B": {"lat": 71.140556, "lon": -51.128333, "alt": 0},
}

interp_method = "linear"  # nearest | linear
height_correction = True

for station in ("WEG_B", "WEG_L"):
    lat = station_coords[station]["lat"]
    lon = station_coords[station]["lon"]

    ifiles_mean = [
        f"{reanalysis_folder}/air.2m/air.2m.{year}.nc" for year in range(2014, 2015 + 1)
    ]
    orog_ncep_file = ".../20Cv3/surface.height/hgt.sfc.SI.nc" # to get surface height
    reana_mean = xr.open_mfdataset(ifiles_mean)["air"]

    orog_ncep = xr.open_dataset(orog_ncep_file)["hgt"]

    # ncep works with 0-360 longitudes
    if lon < 0:
        lon_ncep = 360 + lon
    else:
        lon_ncep = lon
    reana_mean = reana_mean.interp(lat=lat, lon=lon_ncep, method=interp_method)
    orog_ncep = orog_ncep.interp(lat=lat, lon=lon_ncep, method=interp_method)

    reana_mean = reana_mean - 273.15
    reana_mean = reana_mean.load()

    z_model = orog_ncep.squeeze().data
    z_station = station_coords[station]["alt"]
    delta_T = (z_model - z_station) / 1000 * 6.5  # correction with 6.5K/km
    reana_mean = reana_mean + delta_T

    df = reana_mean.to_dataframe(name="value")

    df.drop(df.columns[[0, 1]], axis=1, inplace=True)
    df.columns = ["AT"]
    daily_averages = df.resample("D").mean()

    daily_averages.to_csv(
        ".../20CRv3/AT_20CRv3_{s}_daily.csv".format(s=station)
    )

    df.to_csv(".../20CRv3/AT_20CRv3_{s}.csv".format(s=station))
# %%
for station in ("WEG_L", "WEG_B"):
    df = pd.read_csv(
        ".../20CRv3/AT_20CRv3_{s}.csv".format(s=station)
    )
    df.drop(df.columns[[1, 2]], axis=1, inplace=True)
    df.columns = ["date", "AT"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    daily_averages = df.resample("D").mean()

    daily_averages.to_csv(
        ".../20CRv3/AT_20CRv3_{s}_daily.csv".format(s=station)
    )




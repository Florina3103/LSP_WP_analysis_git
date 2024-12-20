
#%%
import xarray as xr
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sompy import *
import matplotlib.colors as colors
from matplotlib.colors import Normalize
import math

import SOM_class
import importlib
importlib.reload(SOM_class)
import matplotlib as mpl

from matplotlib.dates import HourLocator, DateFormatter, MonthLocator


# %% define parameters for SOM

path = '/mnt/hdd2/users/florina/20CRv3'
domain = 'GRl'
start_year = 1900
end_year = 2015
variables = ['hgt']
sims = ['ssim']
n = 8
iterations = 10000

#%% run SOMs 

SOM = SOM_class.SOM(path, domain, start_year, end_year, variables, sims, n, False, True, iterations)


#%% save dataset with specific parameter for futher analysis

dims = ('n', 'lat', 'lon')  # Example dimensions
coords = {'n': , 'lat': SOM.input['lat'] , 'lon':SOM.input['lon']}  # Example coordinates

# Create xarray dataset from the array
ds = xr.Dataset(
    { 'data': (('n', 'lat', 'lon'), SOM.som['ssim']['hgt']['data'][1]) },  # Data variable
    coords=coords
)

#%%
import os
import xarray as xr

import pandas as pd
from sompy import *
import matplotlib.colors as colors
from matplotlib.colors import Normalize

import SOM_class
import importlib
importlib.reload(SOM_class)
import matplotlib as mpl
import os

from matplotlib.dates import HourLocator, DateFormatter, MonthLocator


# %% define parameters for SOM

path = '.../20CRv3'
domain = 'GRl' # choose domain based on domain_limits_20CRv3.txt

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

ds = ds.assign_attrs(
    description="SOMs of "+variables[0]+" based on " + sims[0],
    start_year=start_year, end_year = end_year,
    domain = domain,
    iterations = iterations,
    n = n
)

path_save_output = '/mnt/hdd2/users/florina/20CRv3/SOM_output/sample_datasets'

subfolder_name = 'SOM_'+str(n)+'_'+sims[0]+'_'+variables[0]+'_' + domain + '_'+str(start_year)+'_'+str(end_year)+'_third_sample'

subfolder_path = os.path.join(path_save_output, subfolder_name)
    
# Check if the subfolder already exists; if not, create it
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)


ds.to_netcdf(path=subfolder_path +'/SOM_'+str(n)+'_'
             +sims[0]+'_'+variables[0]+'_' + domain + '_'
             +str(start_year)+'_'+str(end_year), mode='w',)
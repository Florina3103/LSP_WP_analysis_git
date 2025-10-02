# Robustness test of SOM sorting influence on significant differences in BMU distribution
# Author: Florina Schalamon
#======================
#%%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from sompy import *
import matplotlib.colors as colors
from matplotlib.colors import Normalize
import math
# import xarray_clim as xarray_clim
import SOM_class
import importlib
importlib.reload(SOM_class)
import scipy.stats as stats
from datetime import datetime



import matplotlib as mpl

from matplotlib.dates import HourLocator, DateFormatter, MonthLocator

alpha = 0.05
#%%

def chisquare_test(counts1, counts2):
    """counts1 and counts2 need to be in she same order of variables:
    [count bmu 1, count bmu 2, .......]"""

    if np.min(counts1) < 5 or np.min(counts2)  < 5:
        print('waring: some BMUS occur less often than 5, chi square test is problematic in this case')

    sres = stats.chi2_contingency([counts1, counts2])

    pvalue = sres.pvalue


    # if pvalue<alpha:
    #     print("""Null hypothesis rejected -> BMU distribution and period are not 
    #         independent -> BMU distribution in period 1 and period 2 is significantly different""")
    # else:
    #     print("""Null hypothesis not rejected -> BMU distribution and period are 
    #         independent -> BMU distribution in period 1 and period 2 is not significantly different""")

    distributions_different = pvalue<alpha
    print('pvalue:', pvalue)
    return distributions_different, pvalue


# %% define parameters for SOM

path = '.../20CRv3'
domain = 'GRl' # 'NH_small', 'NH', 'GR_large
start_year = 1900
end_year = 2015
variables = ['hgt']
sims = ['ssim']#['ed', 'str', 'ssim']#['ssim', 'ed','str']
n = 10
iterations = 10000

n_robustness = 1500 # how many times the SOM is run with same parameters to test robustness of results
period1 = pd.to_datetime(['19220101', '19321231'])
period2 = ['19930101', '20071231']

#%% load input data 
input_data = SOM_class.get_input_dataset(path, domain, start_year, end_year, variables[0])

#%% run SOM
var = variables[0]
iput1d = np.array([np.array([vari]).flatten() for vari in zip(input_data[var].values)])

data01 = []
data02 = []
data12 = []

for i in range(n_robustness):
    print(i)
    somout = som(iput1d, n, iterate = iterations, sim=sims[0]) 
    bmu =  np.array(somout['bmu_proj_fin'])

    #create temp bmu datafram
    df_bmu = pd.DataFrame(bmu, columns = ['bmu'])
    df_bmu['bmu'] = df_bmu['bmu'].astype(int) +1
    df_bmu['time'] = input_data['time']

    df_p1 = df_bmu[(df_bmu['time'] >= period1[0]) & (df_bmu['time'] <= period1[1])].copy()
    df_p2 = df_bmu[(df_bmu['time'] >= period2[0]) & (df_bmu['time'] <= period2[1])].copy()

    counts0 = df_bmu['bmu'].value_counts().sort_index()
    counts1 = df_p1['bmu'].value_counts().sort_index()
    counts2 = df_p2['bmu'].value_counts().sort_index()  
    
    # statistical tests
    test01 = chisquare_test(counts0, counts1)
    test02 = chisquare_test(counts0, counts2)
    test12 = chisquare_test(counts1, counts2)

    data01.append((i,test01[0], test01[1]))
    data02.append((i,test02[0], test02[1]))
    data12.append((i,test12[0], test12[1]))
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


df_testing_all_wp1 = pd.DataFrame(data01, columns = ['run', 'h0 rejected', 'pvalue'])
df_testing_all_wp2 = pd.DataFrame(data02, columns = ['run', 'h0 rejected', 'pvalue'])
df_testing_wp1_wp2 = pd.DataFrame(data12, columns = ['run', 'h0 rejected', 'pvalue'])

df_testing_all_wp1.set_index('run', inplace=True)
df_testing_all_wp2.set_index('run', inplace=True)
df_testing_wp1_wp2.set_index('run', inplace=True)

path ='.../SOM_robustness_test/'
df_testing_all_wp1.to_csv(path + f'chi2_test_{n_robustness}_all_wp1_{n}_ssim_{iterations}_hgt_GRl_1900_2015.csv')
df_testing_all_wp2.to_csv(path + f'chi2_test_{n_robustness}_all_wp2_{n}_ssim_{iterations}_hgt_GRl_1900_2015.csv')
df_testing_wp1_wp2.to_csv(path + f'chi2_test_{n_robustness}_wp1_wp2_{n}_ssim_{iterations}_hgt_GRl_1900_2015.csv')
#%% 

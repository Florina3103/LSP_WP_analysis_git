# Self-organizing map
# Author: Florina Schalamon based on Doan, Q. V. (2021). S-SOM v1.0: A structural self-organizing map algorithm for weather typing (Version V1). Zenodo. https://doi.org/10.5281/zenodo.4437954

#======================
#%%
#load necessary packages
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from sompy import *
# import xarray_clim as xarray_clim
import cartopy.crs as ccrs

def weighted_avg_and_var(values, weights):
    """
    Return the weighted average and variance.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, variance)

def bmu_given_nodes_weight(sample, candidate, weight_lat, method="ed"):
    """
    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.
    candidate : TYPE
        DESCRIPTION.
    weight_lat : array
        the weight array in rad
    method : TYPE, optional
        DESCRIPTION. The default is 'ed'.

    Returns
    -------
    bmu_1d : TYPE
        DESCRIPTION.
    smu_1d : TYPE
        DESCRIPTION.
    maxv : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.

    """

    if method == "ssim":
        values = []
        x = sample
        x_mean, x_var = weighted_avg_and_var(x, weight_lat)
        for y in candidate[:]:
            y_mean, y_var = weighted_avg_and_var(y.flatten(), weight_lat)
            term1 = 2 * x_mean * y_mean / (x_mean**2 + y_mean**2)
            term2 = 2 * np.cov(x.flatten(), y.flatten())[1, 0] / (x_var + y_var)
            values.append(term1 * term2)
        values = np.array(values)

    # ====================
    if method == "sc":
        values = []
        x = sample
        x_mean, x_var = weighted_avg_and_var(x, weight_lat)
        for y in candidate[:]:
            y_mean, y_var = weighted_avg_and_var(y.flatten(), weight_lat)
            term1 = 1.0  # 2*x_mean*y.mean() / (x_mean**2 + y.mean()**2)
            term2 = 2 * np.cov(x.flatten(), y.flatten())[1, 0] / (x_var + y_var)
            values.append(term1 * term2)
        values = np.array(values)
    # ====================

    if method == "ed":
        sub = np.sqrt(weight_lat * (candidate - sample) ** 2)
        values = -np.linalg.norm(sub**2, axis=1)
    # ====================

    if method == "lum":
        values = []
        x = sample
        x_mean, x_var = weighted_avg_and_var(x, weight_lat)
        for y in candidate[:]:
            y_mean, y_var = weighted_avg_and_var(y.flatten(), weight_lat)
            values.append(2 * x_mean * y_mean / (x_mean**2 + y_mean**2))
        values = np.array(values)
    # ====================

    if method == "cnt":
        values = []
        x = sample
        x_mean, x_var = weighted_avg_and_var(x, weight_lat)
        for y in candidate[:]:
            y_mean, y_var = weighted_avg_and_var(y.flatten(), weight_lat)
            values.append(2 * np.sqrt(x_var) * np.sqrt(y_var) / (x_var + y_var))
        values = np.array(values)
    # ====================

    if method == "str":
        values = []
        x = sample
        x_mean, x_var = weighted_avg_and_var(x, weight_lat)
        for y in candidate[:]:
            y_mean, y_var = weighted_avg_and_var(y.flatten(), weight_lat)
            values.append(
                np.cov(x.flatten(), y.flatten(), aweights=weight_lat)[1, 0]
                / (np.sqrt(x_var) * np.sqrt(y_var))
            )
        values = np.array(values)

    maxv = np.max(values)
    bmu_1d, smu_1d = np.argsort(values)[-1], np.argsort(values)[-2]
    # print(values)
    return bmu_1d, smu_1d, maxv, values


def get_files_in_timespan(folder_path, start_year, end_year):
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    
    # Initialize an empty list to store filenames within the timespan
    files_in_timespan = []
    
    # Loop through each file to check if it contains a valid year in the name
    for file_name in all_files:
        # Check if the file name has the required format 'hgt.YEAR.nc'
        if file_name.startswith('hgt.') and file_name.endswith('.nc'):
            # Extract the year from the filename
            year_str = file_name.split('.')[-2]
            
            # Check if the extracted part is a valid year (consisting of digits only)
            if year_str.isdigit():
                year = int(year_str)
                
                # Check if the year is within the specified range
                if start_year <= year <= end_year:
                    # Append the filename to the list if it falls within the timespan
                    files_in_timespan.append(file_name)
    
    return files_in_timespan


#=======================


def get_domain_limits(path, domain):
    """
    Method to retrieve the actual model grid points for a certain area as defined in the corresponding text file.
    """

    limits = pd.read_csv(path+'/domain_limits_20CRv3.txt', delim_whitespace=True, dtype=str)
    i = limits["domain"] == domain
    dict_limits = {'x':[int(limits["xmin"][i]), int(limits["xmax"][i])], 'y': [int(limits["ymin"][i]), int(limits["ymax"][i])]}

    return dict_limits

#%%
def get_input_dataset(path, domain, start_year, end_year, var):

            domain_limits = get_domain_limits(path,domain)
            path = path + '/' + var

            file_names = get_files_in_timespan(path, start_year, end_year)
            file_paths = [os.path.join(path, file_name) for file_name in file_names]

            input = xr.open_mfdataset(file_paths, combine="nested", concat_dim='time', data_vars='minimal', coords='minimal')
            
            input = input.isel(lat=slice(domain_limits['y'][0], domain_limits['y'][1]), level = 12) # level 12 is the 500 hpa ... can be adjusted and possibly also as and additional variable

            if domain_limits['x'][0] > domain_limits['x'][1]:
                input = input.sel(lon=slice(domain_limits['x'][0], None)).combine_first(input.sel(lon=slice(None, domain_limits['x'][1])))
            elif domain_limits['x'][0] == 0: 
                input = input.sel(lon=slice(None, domain_limits['x'][1]))
            elif domain_limits['x'][1] == 360: 
                input = input.sel(lon=slice(domain_limits['x'][0], None))
            elif domain_limits['x'][0] > 0: 
                input = input.sel(lon=slice(domain_limits['x'][0], domain_limits['x'][1]))

            input[var].load()

            input['lat'] = input['lat'].astype(dtype='float64')
            input['lon'] = input['lon'].astype(dtype='float64')
           
            lat_1d = input['lat'].values
            lon_1d = input['lon'].values

            # Create a mesh grid for latitudes and longitudes
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

            # Assign the 2D arrays as coordinates to the dataset
            input['lat2d'] = (('lat', 'lon'), lat_2d)
            input['lon2d'] = (('lat', 'lon'), lon_2d)

            return input

#%%
#=======================


class SOM():
     
    def __init__(self,
                 
                    path = '/mnt/hdd2/users/florina/20CRv3', #path where the input data is found
                 
                    domain = "NH", #INPUT which domain should be analysed

                    start_year = 2009,
                    end_year = 2015,   # which years to consider for the analysis 

                    variables = ['hgt'],  # which variables to sort

                    sim = ['ed'],  # which distance functions to use

                    n = 4, # number of nodes 

                    seperate_seasons = True, # boolean to analyse seasons seperatly or not
                    
                    weight_lat = False, # boolean if weighting the nothern grid points accordingly

                    iterate = 5000 # number of iterations for the SOM
           
                 ):
         

        self.path = path 
        self.domain = domain
        self.start_year = start_year
        self.end_year = end_year

        self.variables = variables
        self.sim = sim 
        self.n = n
        self.iterate = iterate
        
        self.seperate_seasons = seperate_seasons
        self.weight_lat = weight_lat
        
            
        self.input = {}
        self.data = {}
        self.som = {}


        for var in self.variables:
            
            path = self.path + '/' + var

            file_names = get_files_in_timespan(path, start_year, end_year)
            file_paths = [os.path.join(path, file_name) for file_name in file_names]

            self.input = xr.open_mfdataset(file_paths, combine="nested", concat_dim='time', data_vars='minimal', coords='minimal')
            self.domain_limits = get_domain_limits(self.path,self.domain)
            self.input = self.input.isel(lat=slice(self.domain_limits['y'][0], self.domain_limits['y'][1]), level = 12) # level 12 is the 500 hpa ... can be adjusted and possibly also as and additional variable

            if self.domain_limits['x'][0] > self.domain_limits['x'][1]:
                self.input = self.input.sel(lon=slice(self.domain_limits['x'][0], None)).combine_first(self.input.sel(lon=slice(None, self.domain_limits['x'][1])))
            elif self.domain_limits['x'][0] == 0: 
                self.input = self.input.sel(lon=slice(None, self.domain_limits['x'][1]))
            elif self.domain_limits['x'][1] == 360: 
                self.input = self.input.sel(lon=slice(self.domain_limits['x'][0], None))
            elif self.domain_limits['x'][0] > 0: 
                self.input = self.input.sel(lon=slice(self.domain_limits['x'][0], self.domain_limits['x'][1]))

            self.input[var].load()

            self.input['lat'] = self.input['lat'].astype(dtype='float64')
            self.input['lon'] = self.input['lon'].astype(dtype='float64')
           
            lat_1d = self.input['lat'].values
            lon_1d = self.input['lon'].values

            # Create a mesh grid for latitudes and longitudes
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

            # Assign the 2D arrays as coordinates to the dataset
            self.input['lat2d'] = (('lat', 'lon'), lat_2d)
            self.input['lon2d'] = (('lat', 'lon'), lon_2d)

            if seperate_seasons: 

                for i,sim in enumerate(self.sim[:]): # two similarity indices
            
                    print(self.n, sim)
                    self.som[sim] = {}
                    self.som[sim][var] = {}
                    self.som_seasonally(var, sim, self.iterate)
                    
            
            else: 

                for i,sim in enumerate(self.sim[:]): # two similarity indices
            
                    print(self.n, sim, self.iterate)
                    self.som[sim] = {}
                    self.som[sim][var] = {}
                    self.som_yearly(var, sim)
                    self.cal_quantilization_error(var, sim)






            
    #=======================

    def som_yearly(self, var, sim):
        
        iput1d = np.array([np.array([vari]).flatten() for vari in zip(self.input[var].values)])
        
        
        iterate = self.iterate # size of 1-D SOM and number of interation
        # run SOM
        if self.weight_lat: 
            iputlat = np.cos(np.radians(np.array(self.input['lat2d']).flatten()))
            somout = som(iput1d,self.n,iterate = self.iterate,sim = sim, iputlat = iputlat, weight_lat = True)
        else:
            somout = som(iput1d, self.n, iterate = iterate, sim=sim) 
        y = somout['som'].reshape(self.n,self.input['lat'].shape[0],self.input['lon'].shape[0])
        self.som[sim][var]['data'] = (['n','lat', 'lon'],  y)
        self.som[sim][var]['bmu'] = (('input'), np.array(somout['bmu_proj_fin']))
    
        return 
#=======================


    def som_seasonally(self,var,sim):

        
        for key in ['MAM','SON', 'DJF', 'JJA'][:]:
            
            self.som[sim][var][key] = {}
            iput1d = np.array([np.array([vari]).flatten() for vari in zip(self.input[var].values)])
            d3 = self.input[var].groupby('time.season')[key]
            # print(key,self.n, sim)
            iterate = self.iterate # size of 1-D SOM and number of interation
            # run SOM
            if self.weight_lat: 
                iputlat = np.cos(np.radians(np.array(self.input['lat2d']).flatten()))
                somout = som(iput1d,self.n,sim = 'ed', iputlat = iputlat, weight_lat = True)
            else:
                somout = som(iput1d, self.n, iterate = iterate, sim=sim) 
            y = somout['som'].reshape(self.n,self.input['lat'].shape[0],self.input['lon'].shape[0])
            self.som[sim][var][key]['data'] = (['n','lat', 'lon'],  y)
            self.som[sim][var][key]['bmu'] = (('input'), np.array(somout['bmu_proj_fin']))
            print('next season')
        return


    def cal_quantilization_error(self, var, sim):

        '''
        The quantization error is the sum of the mean squared differences between all of the daily input data and the nodes to which they best match.
        '''
        self.som[sim][var]['mean_squared_diff'] = {}

        for i,bmu in enumerate(self.som[sim][var]['bmu'][1]):

            self.som[sim][var]['mean_squared_diff'][i] = np.mean((self.input[var].values[i] - self.som[sim][var]['data'][1][bmu])**2)

        return


    def cal_spatial_correlation_nodes(self,var,sim):
        ''' Calculating the spatial correlation between the nodes'''

        self.som[sim][var]['spatial_correlation'] = {}
        average_input = np.array(self.input[var].mean(dim = 'time').values).flatten()
        anom = self.som[sim][var]['data'][1][0].flatten() - average_input
        temp = pd.DataFrame({'node0': anom})

        for i,node in enumerate(self.som[sim][var]['data'][1][1:]):

            temp['node'+str(i+1)] = node.flatten()
        
        self.som[sim][var]['spatial_correlation'] = temp.corr()
        return
    
    def test_add(self):
        print('test')
        return

    def add_second_period(self, year1, year2):
        
        self.input2 = {}

        for var in self.variables:
            
            path = self.path + '/' + var

            file_names = get_files_in_timespan(path, year1, year2)
            file_paths = [os.path.join(path, file_name) for file_name in file_names]

            self.input2 = xr.open_mfdataset(file_paths, combine="nested", concat_dim='time', data_vars='minimal', coords='minimal')
            self.domain_limits = get_domain_limits(self.path,self.domain)
            self.input2 = self.input2.isel(lat=slice(self.domain_limits['y'][0], self.domain_limits['y'][1]), level = 12) # level 12 is the 500 hpa ... can be adjusted and possibly also as and additional variable

            if self.domain_limits['x'][0] > self.domain_limits['x'][1]:
                self.input2 = self.input2.sel(lon=slice(self.domain_limits['x'][0], None)).combine_first(self.input.sel(lon=slice(None, self.domain_limits['x'][1])))
            elif self.domain_limits['x'][0] == 0: 
                self.input2 = self.input2.sel(lon=slice(None, self.domain_limits['x'][1]))
            elif self.domain_limits['x'][1] == 360: 
                self.input2 = self.input2.sel(lon=slice(self.domain_limits['x'][0], None))
            elif self.domain_limits['x'][0] > 0: 
                self.input2 = self.input2.sel(lon=slice(self.domain_limits['x'][0], self.domain_limits['x'][1]))

            self.input2[var].load()

            self.input2['lat'] = self.input2['lat'].astype(dtype='float64')
            self.input2['lon'] = self.input2['lon'].astype(dtype='float64')
           
            lat_1d = self.input2['lat'].values
            lon_1d = self.input2['lon'].values

            # Create a mesh grid for latitudes and longitudes
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

            # Assign the 2D arrays as coordinates to the dataset
            self.input2['lat2d'] = (('lat', 'lon'), lat_2d)
            self.input2['lon2d'] = (('lat', 'lon'), lon_2d)

        return
    
    def bmu_second_period_first_nodes(self):


        for var in self.variables:

            input1d = np.array([np.array([vari]).flatten() for vari in zip(self.input2[var].values)])
            nodes = self.som[self.sim[0]][var]['data'][1].reshape(self.n, self.input2['lat'].shape[0]*self.input2['lon'].shape[0])
    #         if self.weight_lat: 
    #             print('weighted')
            iputlat = np.cos(np.radians(np.array(self.input2['lat2d']).flatten()))
            bmu = [
                            bmu1d_weight(data, nodes, iputlat, method=self.sim[0])[0] 
                            for ind, data in enumerate(input1d)
            ]

        return bmu

   
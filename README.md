The code was used to analyse 20CRv3 data and observations from DMI weather stations. 

The code was used to analyse 20CRv3 data and observations from DMI weather stations for Schalamon et al. 2025. 

	
20CRv3_extract_timeseries_point_location.py - extract timeseries with linear interpolation and height correction 6.5K/1000m to study location

20CRv3_extract_zonal_average.py - extract zonal averages of 20CRv3 with weighted latitude to account for smaller grid cell area in the north, for Greenland, global and Arctic; results see 20CRv3_zonal_mean_annual.csv	

WP_overview.py - script for Figure 2; AT anomaly with respect to 1986-2015 of observations and zonal reanalysis data, as well as spatial representation of gradient in warming periods

SOM_class.py; 	SOM_main.py; 	SOM_visualization.py - scripts for SOM analysis based on Doan et al. 2021 and adjusted for Schalamon et al. 2025; python class, main analysis to define cluster centers and sort input data into them; visualization of the cluster centers (Figure 3)

results SOM: 	
- bmu_SOM_8_ssim_hgt_GRl_1900_2015.csv - into which cluster center the synoptic situation was sorted into per day from 1900-2015
- SOM_8_ssim_hgt_GRl_1900_2015 - the defined cluster centers used for the analysis

significance_tests.py - testing if the difference between cluster occurence is significant between periods

SOM_robustness_test.py - testing if the result is robust and not depending on the choosen clusters; results are in chi2_test_1500_all_... .csv

LSP_occurence_AT_anomaly_persistence.py - script for Figure 4 and 5

LSP_extrem_days.py - script for Figure 6

data - folder containing results of zonal averages, SOM cluster centers and daily cluster center based on SOM analysis, domain limits for the analysis, and results of the robustness test of the significant difference between distribution 

xarray_clim.py is written by Sebastian Scher (seb1000 on github), this package is used for handling climate grid data

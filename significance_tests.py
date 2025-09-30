#%%
import pandas as pd
from pylab import plt 
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import PermutationMethod, MonteCarloMethod

def chisquare_test(counts1, counts2):
    """counts1 and counts2 need to be in she same order of variables:
    [count bmu 1, count bmu 2, .......]"""

    if np.min(counts1) < 5 or np.min(counts2)  < 5:
        print('waring: some BMUS occur less often than 5, chi square test is problematic in this case')

    sres =  stats.chi2_contingency([counts1, counts2])

    pvalue = sres.pvalue


    if pvalue<alpha:
        print("""Null hypothesis rejected -> BMU distribution and period are not 
            independent -> BMU distribution in period 1 and period 2 is significantly different""")
    else:
        print("""Null hypothesis not rejected -> BMU distribution and period are 
            independent -> BMU distribution in period 1 and period 2 is not significantly different""")

    distributions_different = pvalue<alpha

    return distributions_different, pvalue

def g_test(counts1, counts2):
    """counts1 and counts2 need to be in she same order of variables:
    [count bmu 1, count bmu 2, .......]"""

    if np.min(counts1) < 5 or np.min(counts2)  < 5:
        print('waring: some BMUS occur less often than 5, chi square test is problematic in this case')

    sres =  stats.chi2_contingency([counts1, counts2],lambda_="log-likelihood")

    pvalue = sres.pvalue


    if pvalue<alpha:
        print("""Null hypothesis rejected -> BMU distribution and period are not 
            independent -> BMU distribution in period 1 and period 2 is significantly different""")
    else:
        print("""Null hypothesis not rejected -> BMU distribution and period are 
            independent -> BMU distribution in period 1 and period 2 is not significantly different""")

    distributions_different = pvalue<alpha

    return distributions_different, pvalue

def montecarlo_test(counts1, counts2):
    """counts1 and counts2 need to be in she same order of variables:
    [count bmu 1, count bmu 2, .......]"""

    if np.min(counts1) < 5 or np.min(counts2)  < 5:
        print('waring: some BMUS occur less often than 5, chi square test is problematic in this case')

    sres =  stats.chi2_contingency([counts1, counts2], correction=False, method=MonteCarloMethod())

    pvalue = sres.pvalue


    if pvalue<alpha:
        print("""Null hypothesis rejected -> BMU distribution and period are not 
            independent -> BMU distribution in period 1 and period 2 is significantly different""")
    else:
        print("""Null hypothesis not rejected -> BMU distribution and period are 
            independent -> BMU distribution in period 1 and period 2 is not significantly different""")

    distributions_different = pvalue<alpha

    return distributions_different, pvalue

def permutation_test(counts1, counts2):
    """counts1 and counts2 need to be in she same order of variables:
    [count bmu 1, count bmu 2, .......]"""

    if np.min(counts1) < 5 or np.min(counts2)  < 5:
        print('waring: some BMUS occur less often than 5, chi square test is problematic in this case')

    sres =  stats.chi2_contingency([counts1, counts2], correction=False, method=PermutationMethod())

    pvalue = sres.pvalue


    if pvalue<alpha:
        print("""Null hypothesis rejected -> BMU distribution and period are not 
            independent -> BMU distribution in period 1 and period 2 is significantly different""")
    else:
        print("""Null hypothesis not rejected -> BMU distribution and period are 
            independent -> BMU distribution in period 1 and period 2 is not significantly different""")

    distributions_different = pvalue<alpha

    return distributions_different, pvalue

def fisherexakt_test(counts1, counts2):
    """counts1 and counts2 need to be in she same order of variables:
    [count bmu 1, count bmu 2, .......]"""

    if np.min(counts1) < 5 or np.min(counts2)  < 5:
        print('waring: some BMUS occur less often than 5, chi square test is problematic in this case')

    sres =  stats.fisher_exact([counts1, counts2])

    pvalue = sres.pvalue


    if pvalue<alpha:
        print("""Null hypothesis rejected -> BMU distribution and period are not 
            independent -> BMU distribution in period 1 and period 2 is significantly different""")
    else:
        print("""Null hypothesis not rejected -> BMU distribution and period are 
            independent -> BMU distribution in period 1 and period 2 is not significantly different""")

    distributions_different = pvalue<alpha

    return distributions_different, pvalue 



#%%

n_clusters = 8

df = pd.read_csv(f'/home/flo/LSP_analysis/Data/SOM_{n_clusters}_ssim_hgt_GRl_1900_2015/bmu_SOM_{n_clusters}_ssim_hgt_GRl_1900_2015.csv', index_col=0)
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month

print(df)

period1 = pd.to_datetime(['19220101', '19321231'])
period2 = ['19930101', '20071231']

# sort according to dates

df = df.sort_values('time').reset_index(drop=False)

df_p1 = df[(df['time'] >= period1[0]) & (df['time'] <= period1[1])].copy()
df_p2 = df[(df['time'] >= period2[0]) & (df['time'] <= period2[1])].copy()


alpha = 0.05

counts0 = df['bmu'].value_counts().sort_index()
counts1 = df_p1['bmu'].value_counts().sort_index()
counts2 = df_p2['bmu'].value_counts().sort_index()

df_pvalues = pd.DataFrame(columns=['method', 'WP1 vs WP2', 'WP1 vs all', 'WP2 vs all'])
#%% chi square test

print('chi square test')
print('testing WP1 vs WP2')
res,p_value1 = chisquare_test(counts1, counts2)

print('testing WP1 vs all')
res, p_value2 =  chisquare_test(counts1, counts0)

print('testing WP2 vs all')
res, p_value3=  chisquare_test(counts2, counts0)

new_row = pd.DataFrame([{'method': 'chi square', 
                         'WP1 vs WP2': p_value1, 
                         'WP1 vs all': p_value2, 
                         'WP2 vs all': p_value3}])

df_pvalues = pd.concat([df_pvalues, new_row], ignore_index=True)
#%% g-test

print('g test')
print('testing WP1 vs WP2 G-test')
res, p_value1 =  g_test(counts1, counts2)

print('testing WP1 vs all G-test')
res, p_value2 =  g_test(counts1, counts0)

print('testing WP2 vs all G-test')
res, p_value3 =  g_test(counts2, counts0)

new_row = pd.DataFrame([{'method': 'g test',
                         'WP1 vs WP2': p_value1, 
                         'WP1 vs all': p_value2, 
                         'WP2 vs all': p_value3}])

df_pvalues = pd.concat([df_pvalues, new_row], ignore_index=True)

#%% montecarlo
print('montecarlo')

print('testing WP1 vs WP2 montecarlo')
res, p_value1 =  montecarlo_test(counts1, counts2)

print('testing WP1 vs all montecarlo')
res, p_value2 =  montecarlo_test(counts1, counts0)

print('testing WP2 vs all montecarlo')
res, p_value3 =  montecarlo_test(counts2, counts0)

new_row = pd.DataFrame([{'method': 'montecarlo',
                         'WP1 vs WP2': p_value1, 
                         'WP1 vs all': p_value2, 
                         'WP2 vs all': p_value3}])

df_pvalues = pd.concat([df_pvalues, new_row], ignore_index=True)

#%% permutation
print('permutation')

print('testing WP1 vs WP2 permutation')
res, p_value1 =  permutation_test(counts1, counts2)

print('testing WP1 vs all permutation')
res, p_value2 =  permutation_test(counts1, counts0)

print('testing WP2 vs all permutation')
res, p_value3 =  permutation_test(counts2, counts0)

new_row = pd.DataFrame([{'method': 'permutation',
                         'WP1 vs WP2': p_value1, 
                         'WP1 vs all': p_value2, 
                         'WP2 vs all': p_value3}])

df_pvalues = pd.concat([df_pvalues, new_row], ignore_index=True)

# %% fisher

print('testing WP1 vs WP2 fisher')
res, p_value1 =  fisherexakt_test(counts1, counts2)

print('testing WP1 vs all fisher')
res, p_value2 =  fisherexakt_test(counts1, counts0)

print('testing WP2 vs all fisher')
res, p_value3 =  fisherexakt_test(counts2, counts0)

new_row = pd.DataFrame([{'method': 'fisher',
                         'WP1 vs WP2': p_value1, 
                         'WP1 vs all': p_value2, 
                         'WP2 vs all': p_value3}])

df_pvalues = pd.concat([df_pvalues, new_row], ignore_index=True)
# %%

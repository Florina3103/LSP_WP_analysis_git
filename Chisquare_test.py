#%%
import pandas as pd
from pylab import plt 
import seaborn as sns
import numpy as np
import scipy.stats as stats


def chisquare_test(counts1, counts2):
    """counts1 and counts2 need to be in she same order of variables:
    [count bmu 1, count bmu 2, .......]"""

    if np.min(counts1) < 5 or np.min(counts2)  < 5:
        print('waring: some BMUS occur less often than 5, chi square test is problematic in this case')

    sres = stats.chi2_contingency([counts1, counts2])

    pvalue = sres.pvalue


    if pvalue<alpha:
        print("""Null hypothesis rejected -> BMU distribution and period are not 
            independent -> BMU distribution in period 1 and period 2 is significantly different""")
    else:
        print("""Null hypothesis not rejected -> BMU distribution and period are 
            independent -> BMU distribution in period 1 and period 2 is not significantly different""")

    distributions_different = pvalue<alpha

    return distributions_different


#%%

n_clusters = 8

df = pd.read_csv('/home/flo/LSP_analysis/LSP_manuscript_code/bmu_SOM_8_ssim_hgt_GRl_1900_2015.csv', index_col=0)
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month
df['season'] = df['month'].apply(get_season)
print(df)

period1 = pd.to_datetime(['19220101', '19321231'])
period2 = ['19930101', '20071231']

# sort according to dates

df = df.sort_values('time').reset_index(drop=False)

df_p1 = df[(df['time'] >= period1[0]) & (df['time'] <= period1[1])].copy()
df_p2 = df[(df['time'] >= period2[0]) & (df['time'] <= period2[1])].copy()


# statistical tests

# note that we can represent our problem as a 2 x nclusters contingency table
# the easiest (but not necessarily best) solution is to use a chi square test
# in this case, we have 2 variables: period, and bmu, and we test whether the two are independent of each other


# other potential options
# https://en.wikipedia.org/wiki/Cohen%27s_kappa
# https://stats.stackexchange.com/questions/492223/how-to-compare-if-two-multinomial-distributions-are-significantly-different
# https://arxiv.org/pdf/2305.08609.pdf
# https://www.tandfonline.com/doi/full/10.1080/02664763.2019.1601689

# pairwise comparisons https://link.springer.com/content/pdf/10.1198/108571101317096532.pdf




alpha = 0.05

counts0 = df['bmu'].value_counts().sort_index()
counts1 = df_p1['bmu'].value_counts().sort_index()
counts2 = df_p2['bmu'].value_counts().sort_index()




print('testing WP1 vs WP2')
res = chisquare_test(counts1, counts2)

print('testing WP1 vs all')
res = chisquare_test(counts1, counts0)

print('testing WP2 vs all')
res = chisquare_test(counts2, counts0)
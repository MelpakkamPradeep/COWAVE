"""
@author: melpakkampradeep
"""

# Import required libraries
import numpy
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import statsmodels.nonparametric.smoothers_lowess as sm
from scipy.optimize import fmin

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.figure(figsize=(15,8))
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)

# Read dataset (.csv format)
datafull = pd.read_csv("WHO-COVID-19-global-data.csv")
datafull = datafull.drop(columns=['Country', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths'])

x = 'EG'
data = datafull[datafull.Country_code == x]
# Seperate out the required measure.
country = data.loc[:, 'New_cases']
country = country.to_numpy()
country = country.flatten('C')
country = country[0:656]
csize = country.size

# Compute doubling rate
doubling_rate = np.zeros((1, csize))
for i in range(10, csize, 1):
    if(country[i] != 0 and country[i - 7] != 0):
        doubling_rate[0][i] = math.log2(abs(country[i]/country[i - 7]))

doubling_rate = doubling_rate[0]

# Compute doubling time
doubling_time = np.zeros((1, doubling_rate.size))

doubling_rate_norm = doubling_rate/np.amax(doubling_rate)

# Normalize country data
countrynorm = country/np.amax(country)

# Plot country data
plt.plot(countrynorm, label="Raw Data")

# LOESS may change graph significantly, so use right frac param

# Apply LOESS smoothening to country data
x = list(np.arange(1, countrynorm.size + 1))
countrynorm = sm.lowess(countrynorm, x, frac=1/20)
countrynorm = countrynorm[:, 1]

minima = np.zeros((1000, 1))
maxima = np.zeros((1000, 1))
mindex = 1
maxdex = 1

for i in range(2, countrynorm.size - 1):
    if countrynorm[i - 1] > countrynorm[i] and countrynorm[i + 1] > countrynorm[i]:
        minima[mindex] = i
        mindex = mindex + 1

for i in range(2, countrynorm.size - 1):
    if countrynorm[i - 1] < countrynorm[i] and countrynorm[i + 1] < countrynorm[i]:
        maxima[maxdex] = i
        maxdex = maxdex + 1

minima = minima.astype(int)
maxima = maxima.astype(int)

zmin = numpy.count_nonzero(minima) + 2
zmax = numpy.count_nonzero(maxima) + 2
minima = minima[0:zmin]
maxima = maxima[0:zmax]
minima[0] = 1
maxima[0] = 1

# Apply LOESS smoothening to doubling rate norm data

doubling_rate_norm = sm.lowess(doubling_rate_norm, x, frac=1/20) # Change to very low number to generate Fig. 6
doubling_rate_norm = doubling_rate_norm[:, 1]
plt.plot(doubling_rate_norm, label="Doubling Rate")
plt.plot(countrynorm, label="Smoothed Data")

plt.hlines(0, 0, 660)

# Find roots (They give the maximas and minimas of the wave)
roots = np.zeros((1000, 1))
rootdex = 1

for i in range(1, doubling_rate_norm.size - 1):
    if doubling_rate_norm[i]*doubling_rate_norm[i + 1] < 0:
        roots[rootdex] = i
        rootdex = rootdex + 1

roots = roots.astype(int)

zroots = numpy.count_nonzero(roots) + 2
roots = roots[0:zroots]
roots[0] = 1


wave = np.zeros((doubling_rate_norm.size, 1))

for i in range(doubling_rate_norm.size):
    if abs(doubling_rate_norm[i]) >= 0.01:
        wave[i] = 1

plt.plot(wave, label="Wave")

plt.legend()
plt.show()

"""
@author: melpakkampradeep
"""

# Import required libraries
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import statsmodels.nonparametric.smoothers_lowess as sm
from statsmodels.tsa.api import SimpleExpSmoothing

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

pd.set_option('max_colwidth', None) # show full width of showing cols
pd.set_option("expand_frame_repr", False) # print cols side by side as it's supposed to be

dataset = pd.DataFrame()

# Read dataset (.csv format)
datafull = pd.read_csv("WHO-COVID-19-global-data.csv")
datafull = datafull.drop(columns=['Country', 'Cumulative_cases', 'New_cases', 'Cumulative_deaths'])

countrylist = np.array(['IT', 'IN', 'NL', 'EG'])
# Iterate through countries in dataset
for c in countrylist[0:countrylist.size]:
    plt.figure(figsize=(12, 8))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    data = datafull[datafull.Country_code == c]
    # Seperate out the required measure.
    country = data.loc[:, 'New_deaths']

    # Convert to numpy array
    country = country.to_numpy()

    # Flatten array
    country = country.flatten('C')
    country = country[0:656]
    csize = country.size

    # Mean Normalize country data
    if(np.amax(country) > 0):
        countrynorm = (country - np.average(country))/(np.amax(country) - np.amin(country))
    else:
        continue

    plt.plot(countrynorm, label="Normalized Data")

    # Exponential smooth country data
    countrynormexp = SimpleExpSmoothing(countrynorm, initialization_method="estimated").fit()
    countrynorm = countrynormexp.fittedvalues

    # LOESS may change graph significantly, so use right frac param

    # Apply LOESS smoothening to country data
    x = list(np.arange(1, countrynorm.size + 1))
    countrynorm = sm.lowess(countrynorm, x, frac=1/14)
    countrynorm = countrynorm[:, 1]

    plt.plot(countrynorm, label="Smoothed Data")

    # wave array to store if wave or not, index to store the days when a waves starts/ends in order for the correction factor, less than 8 waves are assumed to have occurred
    x_num_of_waves = 8
    wave = np.zeros((countrynorm.size, 1))
    index = np.zeros((2 * x_num_of_waves, 1))
    indexk = 1

    # wave or not
    for i in range(countrynorm.size):
        if countrynorm[i] >= 0:
            wave[i] = 1
        else:
            wave[i] = 0

    flag = 0

    # To capture more of the wave, this correction factor is used (hyperparam)
    correction_factor = 6

    for i in range(wave.size):
        if(wave[i] == 1 and flag == 0):
            index[indexk] = i
            flag = 1
            indexk = indexk + 1
        if(wave[i] == 0 and flag == 1):
            flag = 0
            index[indexk] = i
            indexk = indexk + 1

    index = index[0:np.count_nonzero(index) + 1]

    # Apply correction to wave array
    for i in range(0, index.size - 1, 1):
        wavelength = index[i + 1] - index[i]
        wavelength = math.floor(wavelength/correction_factor)
        if(wavelength <= index[i]):
            for j in range(wavelength):
                wave[int(index[i]) - j] = 1

    wave = wave.astype(int)

    plt.plot(wave, label="Wave")
    plt.legend()

plt.show()





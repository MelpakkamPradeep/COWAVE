"""
@author: melpakkampradeep
"""

# Import required libraries
import pandas as pd
import csv
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
import statsmodels.nonparametric.smoothers_lowess as sm
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox
from scipy.stats import entropy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Set display options

# Show maximum possible column width
pd.set_option('max_colwidth', None) 

# Print columns side by side, as in DataFrame
pd.set_option("expand_frame_repr", False) 

# See README file for general information
def labeller_1(WHO_Data):
    dataset = pd.DataFrame()
    # Read dataset (.csv format)
    datafull = WHO_Data
    
    # Consider only new cases data, can be manually changed for new deaths data as well
    datafull = datafull.drop(columns=['Country', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths', 'WHO_region'])

    # Make list of countries present
    countrylist = datafull.loc[:, 'Country_code'].unique()
    
    # Iterate through countries in dataset
    for c in countrylist[0:countrylist.size]:
        data = datafull[datafull.Country_code == c]
        
        # Separate out the required measure
        country = data.loc[:, 'New_cases']

        # Convert to numpy array
        country = country.to_numpy()

        # Flatten array
        country = country.flatten('C')
        csize = country.size

        # Mean Normalize country data
        if(np.amax(country) > 0):
            countrynorm = (country - np.average(country))/(np.amax(country) - np.amin(country))
        else:
            continue

        # Exponential smooth country data
        countrynormexp = SimpleExpSmoothing(countrynorm, initialization_method="estimated").fit()
        countrynorm = countrynormexp.fittedvalues

        # LOWESS may change graph significantly, so use right frac param

        # Apply LOWESS smoothening to country data
        x = list(np.arange(1, countrynorm.size + 1))
        
        # LOWESS parameter used is 1/14
        countrynorm = sm.lowess(countrynorm, x, frac=1/14)
        countrynorm = countrynorm[:, 1]


        # wave array to store if wave or not, index to store the days when a waves starts/ends in order for the correction factor, less than 128 waves are assumed to have occurred
        x_num_of_waves = 128
        wave = np.zeros((countrynorm.size, 1))
        index = np.zeros((2 * x_num_of_waves, 1))
        indexk = 1

        # Labeling Rule
        for i in range(countrynorm.size):
            if countrynorm[i] >= 0:
                wave[i] = 1
            else:
                wave[i] = 0

        flag = 0

        # To capture more of the wave, this correction factor is used, can be tweaked to obtain a better wave definition
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
        data = data.assign(Wave = wave)
        dataset = dataset.append(data)
        
    # Convert to .csv file and save dataset
    #dataset_out = dataset.to_csv('COVID19_dataset_v1.csv', index=False)

    return dataset

# See README for general information
def labeller_2(WHO_Data):
    datafull = WHO_Data
    
    # As more days pass, the number of rows in temp may have to be increased from 1000000.
    temp = np.zeros((1000000, 4))
    
    # Various fields in the dataset
    dataset = pd.DataFrame({'Date': temp[:, 0], 'Country_code': temp[:, 1], 'Wave': temp[:, 2], 'Cases': temp[:, 3]})
    dataset['Date'] = dataset['Date'].astype(object)
    dataset['Cases'] = dataset['Cases'].astype(object)
    dataset['Country_code'] = dataset['Country_code'].astype(str)
    
    # List of countries
    countrylist = datafull.loc[:, 'Country_code'].unique()

    k = 0
    length = 0

    for c in countrylist[0:countrylist.size]:
        data = datafull[datafull.Country_code == c]

        # Separate out the required measure
        country = data.loc[:, 'New_cases']
        wavenum = data.loc[:, 'Wave']
        code = data.loc[:, 'Country_code']
        date = data.loc[:, 'Date_reported']

        # Convert to numpy array
        country = country.to_numpy()
        wavenum = wavenum.to_numpy()
        date = date.to_numpy()

        # Flatten array
        country = country.flatten('C')
        csize = country.size
        wavenum = wavenum.flatten('C')
        wsize = wavenum.size
        date = date.flatten('C')
        dsize = date.size
        
        # Fill general information
        caselist = [country[0]]
        dataset.iat[k, 0] = date[0]
        dataset.iat[k, 1] = c
        dataset.iat[k, 2] = wavenum[0]

	# If both days belong to the same wave, append to the list, then fill general information.
        for j in range(0, wavenum.size - 1, 1):
            if wavenum[j] == wavenum[j + 1]:
                caselist.append(country[j])
            else:
                caselist.append(country[j])
                dataset.iat[k, 3] = caselist
                length = length + len(caselist)
                caselist = []
                k = k + 1
                dataset.iat[k, 0] = date[j + 1]
                dataset.iat[k, 1] = c
                dataset.iat[k, 2] = wavenum[j + 1]

	# Append list of cases during wave/non-wave to dataset
        dataset.iat[k, 3] = caselist
        length = length + len(caselist)
        caselist = []
        k = k + 1

    # Remove all rows that contain only zeros
    (row, col) = dataset.shape
    for i in range(row):
        flag = 0
        for j in range(col):
            if dataset.iloc[i, j] == 0:
                flag = flag + 1
        if flag == col - 1:
            index = i
            break

    dataset = dataset.iloc[0:index, :]
    
    # Convert to .csv file and save dataset
    dataset_out = dataset.to_csv('COVID19_dataset_v2.csv', index=False)

    return dataset

# See README for general information
def feature_gen(WHO_Data):
    datafull = WHO_Data
    dataf = pd.DataFrame()
    
    # List of countries. Recommended to generate features only for selected countries. So, manually enter countrylist as a list of countries if possible.
    countrylist = datafull.loc[:, 'Country_code'].unique()

    # Length of initial set of features
    lenx= 24
    
    # The number of rows may have to be increased, as more data is available
    sevenday = np.zeros((750081, lenx))  # 24
    sevenday = pd.DataFrame({'Date': sevenday[:, 0],
                             'Country_code': sevenday[:, 1],
                             'T1': sevenday[:, 2],
                             'T2': sevenday[:, 3],
                             'T3': sevenday[:, 4],
                             'T4': sevenday[:, 5],
                             'T5': sevenday[:, 6],
                             'T6': sevenday[:, 7],
                             'T7': sevenday[:, 8],
                             'T8': sevenday[:, 9],
                             'T9': sevenday[:, 10],
                             'T10': sevenday[:, 11],
                             'T11': sevenday[:, 12],
                             'T12': sevenday[:, 13],
                             'T13': sevenday[:, 14],
                             'T14': sevenday[:, 15],
                             'T15': sevenday[:, 16],
                             'T16': sevenday[:, 17],
                             'T17': sevenday[:, 18],
                             'T18': sevenday[:, 19],
                             'T19': sevenday[:, 20],
                             'T20': sevenday[:, 21],
                             'T21': sevenday[:, 22],
                             'Wave': sevenday[:, 23]})

    k = 0

    # Iterate through countries in dataset
    for c in countrylist[0:countrylist.size]:
        data = datafull[datafull.Country_code == c]

        # Separate out the required measure
        country = data.loc[:, 'New_cases']
        date = data.loc[:, 'Date_reported']
        wavenum = data.loc[:, 'Wave']
        code = data.loc[:, 'Country_code']

        # Convert to numpy array
        country = country.to_numpy()
        wavenum = wavenum.to_numpy()
        date = date.to_numpy()

        # Flatten array
        country = country.flatten('C')
        csize = country.size
        wavenum = wavenum.flatten('C')
        wsize = wavenum.size

	# Convert each 21 day span to a data vector of size 21
        for i in range(int(lenx) - 4, country.size, 1):
            for j in range(2, lenx - 1, 1):
                sevenday.iloc[k, j] = country[i + j - lenx + 2]
            sevenday.iloc[k, lenx - 1] = wavenum[i]
            sevenday.iloc[k, 1] = c
            sevenday.iloc[k, 0] = date[i]
            k = k + 1
            
    # Remove all zero rows
    (row, col) = sevenday.shape
    for i in range(row):
        flag = 0
        for j in range(col):
            if sevenday.iloc[i, j] == 0:
                flag = flag + 1
        if flag == col:
            index = i
            break

    sevenday = sevenday.iloc[0:index, :]
    
    # Dataf is the final dataset
    dataf = sevenday
    
    # Datafull is used to generate the features
    datafull = sevenday
    datafull = datafull.drop(columns=["Date"])

    # Features based on time series decomposition into trend, seasonal and residual components
    r = []
    t = []
    s = []
    for c in countrylist[0:countrylist.size]:
        d = datafull[datafull.Country_code == c]
        data = d.loc[:, 'T21']

        result = seasonal_decompose(data, model='additive', period=21)

        residuals = result.resid.values
        trends = result.trend.values
        residuals[0:10] = 0
        trends[0:10] = 0

        for i in range(residuals.size - 10, residuals.size, 1):
            residuals[i] = (residuals[i - 1] + residuals[i - 2] + residuals[i - 3] + residuals[i - 4] + residuals[
                i - 5] + residuals[i - 6] + residuals[i - 7]) / 7

        for i in range(trends.size - 10, trends.size, 1):
            trends[i] = (trends[i - 1] + trends[i - 2] + trends[i - 3] + trends[i - 4] + trends[i - 5] + trends[i - 6] +
                         trends[i - 7]) / 7

        seasonal = result.seasonal.values

        for i in range(residuals.size):
            r.append(residuals[i])
            s.append(seasonal[i])
            t.append(trends[i])

    # Add decomposition features
    datafull['Residual'] = r
    datafull['Seasonal'] = s
    datafull['Trend'] = t

    dataf['Residual'] = r
    dataf['Seasonal'] = s
    dataf['Trend'] = t

    # Drop newly generated features
    datasmall = datafull.drop(columns=['Country_code', 'Trend', 'Residual', 'Seasonal', 'Wave'])

    # Mean, Variance, PDF and LogReg feature generation
    mean = []
    var = []
    PDF = []
    logreg = []

    for i in range(datafull.loc[:, 'T21'].size):
        data = datasmall.iloc[i, :]
        data.to_numpy()

        m = np.mean(data)
        mean.append(m)

        v = np.var(data)
        var.append(v)

        if np.std(data) > 0:
            y = data[20]/np.std(data)
        else:
            y = 0
        z = 1 / (1 + math.exp(-y))
        logreg.append(z)

        m = np.mean(data)
        std = np.std(data)
        for j in range(data.size):
            scale = data[j] - m  # Change data[i] to data[20]
        if std != 0:
            fac = (1 / (math.sqrt(2 * math.pi) * std))
            norm = (-math.pow(scale, 2)) / (2 * math.pow(std, 2))
            z1 = fac * (math.exp(norm))
        else:
            z1 = 0
        PDF.append(z1)

    # Power Transformation features generation
    # Box-Cox transformation
    bcox = []
    j = 0
    # Iterate through countries in dataset
    for c in countrylist[0:countrylist.size]:
        data = datafull[datafull.Country_code == c]
        data = data.loc[:, 'T21']
        data.to_numpy()
        bc = np.zeros((data.size, 1))
        data[data <= 0] = 1e-100
        bc = boxcox(data)
        bc = bc[0]
        for i in range(len(bc)):
            bcox.append(bc[i])

    # Square root, Square, Log tranformations
    sqroot = []
    sq = []
    lg = []

    data = datafull.loc[:, 'T21']
    for i in range(data.size):
        sqroot.append(math.sqrt(abs(data[i])))
        sq.append(math.pow(data[i], 2))
        if data[i] > 0:
            lg.append(math.log(data[i]))
        else:
            lg.append(0)

    # Generate Minimum and Maximum features
    MAX = []
    MIN = []
    for i in range(datafull.loc[:, 'T21'].size):
        data = datasmall.iloc[i, :]
        data.to_numpy()
        MAX.append(np.amax(data))
        MIN.append(np.amin(data))

    # Append generated features to temporary dataset
    datafull['Box_Cox'] = bcox
    datafull['Sqroot'] = sqroot
    datafull['Sq'] = sq
    datafull['Log'] = lg
    datafull['PDF'] = PDF
    datafull['LogReg'] = logreg
    datafull['MAX'] = MAX
    datafull['MIN'] = MIN
    datafull['Mean'] = mean
    datafull['Variance'] = var

    # Generate difference features
    DIFF = np.zeros((datafull.loc[:, 'T21'].size, 7))
    for i in range(7):
        DIFF[:, i] = datafull.iloc[:, 15 + i] - datafull.iloc[:, 14 + i]

    # Append generated features to temporary dataset
    datafull['D1'] = DIFF[:, 0]
    datafull['D2'] = DIFF[:, 1]
    datafull['D3'] = DIFF[:, 2]
    datafull['D4'] = DIFF[:, 3]
    datafull['D5'] = DIFF[:, 4]
    datafull['D6'] = DIFF[:, 5]
    datafull['D7'] = DIFF[:, 6]

    # Generate Covariance, Median, Range and Entropy features
    CV = []
    for i in range(datafull.loc[:, 'T21'].size):
        if datafull.loc[i, 'Mean'] != 0:
            CV.append(np.sqrt(datafull.loc[i, 'Variance']) / datafull.loc[i, 'Mean'])
        else:
            CV.append(0)

    Med = []
    for i in range(datafull.loc[:, 'T21'].size):
        M = np.median(datafull.iloc[i, 1:lenx-1])
        Med.append(M)

    Range = datafull.loc[:, 'MAX'] - datafull.loc[:, 'MIN']

    Ent = np.zeros((datafull.loc[:, 'T21'].size, 1))
    for i in range(datafull.loc[:, 'T21'].size):
        e = 0
        if max(datafull.iloc[i, 1:lenx-1]) != 0:
            norm = datafull.iloc[i, 1:lenx-1] / max(datafull.iloc[i, 1:lenx-1])
        else:
            norm = np.ones((lenx-1, 1))
        for j in range(norm.size):
            if norm[j] != 0:
                e = e - norm[j] * np.log(abs(norm[j]))
            Ent[i] = e

    # Append all generated features to the dataset
    dataf['Box_Cox'] = bcox
    dataf['Sqroot'] = sqroot
    dataf['Sq'] = sq
    dataf['Log'] = lg
    dataf['PDF'] = PDF
    dataf['LogReg'] = logreg
    dataf['MAX'] = MAX
    dataf['MIN'] = MIN
    dataf['Mean'] = mean
    dataf['Variance'] = var
    dataf['D1'] = DIFF[:, 0]
    dataf['D2'] = DIFF[:, 1]
    dataf['D3'] = DIFF[:, 2]
    dataf['D4'] = DIFF[:, 3]
    dataf['D5'] = DIFF[:, 4]
    dataf['D6'] = DIFF[:, 5]
    dataf['D7'] = DIFF[:, 6]
    dataf['CV'] = CV
    dataf['Median'] = Med
    dataf['Range'] = Range
    dataf['Entropy'] = Ent
    
    # Convert to .csv file and save dataset
    dataset_out = dataf.to_csv('COVID19_dataset_v3.csv', index=False)
    return dataf


# Read data from WHO website, na_filter disabled to account for Namibia's country code
file = pd.read_csv("https://covid19.who.int/WHO-COVID-19-global-data.csv", na_filter=False)

# Uncomment next line if looking to reproduce the data in the paper
#file = pd.read_csv("WHO-COVID-19-global-data.csv", na_filter = False)

# Generate all datasets presented in the functions
datas_1 = labeller_1(file)

datas_2 = labeller_2(datas_1)

file_1 = pd.read_csv("COVID19_dataset_v1.csv", na_filter = False)

datas_3 = feature_gen(file_1)






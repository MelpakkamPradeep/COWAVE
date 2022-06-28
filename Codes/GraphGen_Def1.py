"""
@author: melpakkampradeep
"""
import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.nonparametric.smoothers_lowess as sm

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Read dataset (.csv format)
datafull = pd.read_csv("WHO-COVID-19-global-data.csv")
datafull = datafull.drop(columns=['Country', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths'])

# Make list of countries present
countrylist = datafull.loc[:, 'Country_code'].unique()
countrylist = ['IT', 'IN', 'NL', 'EG']

for c in countrylist:
	data = datafull[datafull.Country_code == c]
	# Seperate out the required measure.
	country = data.loc[:, 'New_cases']
	country = country.to_numpy()
	country = country.flatten('C')
	country = country[0:656]

	#plt.plot(country, label="Raw Data")

	x = list(np.arange(1, country.size + 1))
	country = sm.lowess(country, x, frac=1/20)

	minima = np.zeros((1000, 1))
	maxima = np.zeros((1000, 1))
	mindex = 1
	maxdex = 1

	for i in range(2, country[:, 1].size - 1):
		if country[i - 1, 1] > country[i, 1] and country[i + 1, 1] > country[i, 1]:
			minima[mindex] = country[i, 0]
			mindex = mindex + 1

	for i in range(2, country[:, 1].size - 1):
		if country[i - 1, 1] < country[i, 1] and country[i + 1, 1] < country[i, 1]:
			maxima[maxdex] = country[i, 0]
			maxdex = maxdex + 1

	minima = minima.astype(int)
	maxima = maxima.astype(int)
	zmin = numpy.count_nonzero(minima) + 2
	zmax = numpy.count_nonzero(maxima) + 2
	minima = minima[0:zmin]
	maxima = maxima[0:zmax]
	minima[0] = 1
	maxima[0] = 1

	plt.plot(country[:, 0], country[:, 1], label="Smoothed data")
	#plt.plot(countrydaily[0])
	
	colors=['b', 'r', 'c', 'm', 'y', 'k']
	for i in range(minima.size - 1):
		if(abs(minima[i + 1] - minima[i]) >= 20):
			#Since country[:, 0] starts from 1 instead of 0, the entire plot is shifted back by 1 unit
			for j in range(int(country[minima[i - 1] - 1, 0]), int(country[minima[i] - 1, 0])):
				plt.vlines(j+1, 0, country[j, 1], colors=colors[i % 6])
		for j in range(int(country[minima[i] - 1, 0]), int(country[-1, 0])):
			plt.vlines(j, 0, country[j, 1], colors=colors[(i+1) % 6])
	
	plt.legend()
	plt.show()




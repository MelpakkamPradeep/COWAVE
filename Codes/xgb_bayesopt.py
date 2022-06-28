# -*- coding: utf-8 -*-
"""XGB_BayesOpt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sh2lsc1p1JkaT9pyrlQ3IsTo57WSSY6M
"""
"""
Created on Jun 4, 2022 7:44 AM
@author: melpakkampradeep
"""

!pip install bayesian-optimization

# Import required libraries
import pandas as pd
import numpy as np
import io
import matplotlib
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
import pickle
import bayes_opt
from bayes_opt import BayesianOptimization

from sklearn.experimental import enable_halving_search_cv
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

from google.colab import files

pd.set_option("max_columns", None) # show all cols
pd.set_option('max_colwidth', None) # show full width of showing cols
pd.set_option("expand_frame_repr", False) # print cols side by side as it's supposed to be

import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

uploaded = files.upload()
# Read dataset (.csv format)
datafull = pd.read_csv(io.BytesIO(uploaded['COVID19_dataset_v4.csv']))

# Keep all features but Wave, Date and Country_code
X_t = datafull.drop(columns=['Wave', 'Date', 'Country_code'])

# The target variable
y_t = datafull[['Wave']]

# Train-test split of the data
X_train = X_t.iloc[0:149800]
X_test = X_t.iloc[149801:-1]

y_train = y_t.iloc[0:149800]
y_test = y_t.iloc[149801:-1]

# Train the XGBoost model with the default hyperparameters

# Init classifier
xgb_cl = xgb.XGBClassifier()

# Fit
xgb_cl.fit(X_train, y_train)

# Predict
preds = xgb_cl.predict(X_test)
predst = xgb_cl.predict(X_train)

# Score
print("Test acc: ", accuracy_score(y_test, preds))
print("Test rec:", recall_score(y_test, preds))
print("Test pres:", precision_score(y_test, preds))
print()
print("Train acc: ", accuracy_score(y_train, predst))
print("Train rec:", recall_score(y_train, predst))
print("Train pres:", precision_score(y_train, predst))

# Test-train features chosen to generate n-day vector results
X_train = X_train[['T15','T16','T17','T18','T19','T20','T21']]
X_test = X_test[['T15','T16','T17','T18','T19','T20','T21']]

# Feature Selection

# Define the model
model = XGBClassifier()
# Fit the model
model.fit(X_train, y_train)
# Get importance
importance = model.feature_importances_
# Summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# Plot feature importance
plt.figure(figsize=(30, 5))
plt.plot(X_train.columns, np.transpose(importance))
plt.show()

# Test-train features based on feature selection ("Top 13")
X_train = X_train[['MIN','Range','Sq', 'Median', 'Mean', 'Variance', 'MAX', 'PDF', 'Trend', 'Seasonal', 'Residual', 'T21', 'D7' ]]
X_test = X_test[['MIN','Range','Sq', 'Median', 'Mean', 'Variance', 'MAX', 'PDF', 'Trend', 'Seasonal', 'Residual', 'T21', 'D7' ]]

# Hyperparameter search space
params_xgb = {
  'learning_rate' : (0.0005, 1),
  'max_depth' : (1, 10),
  'min_child_weight' : (1, 10),
  'gamma': (0, 3),
  'colsample_bytree' : (0.001, 1),
  'num_boost_round': (100, 500),
  'reg_lambda': (0.01, 10),
  'scale_pos_weight' : (1, 10),
  'subsample' : (0.001, 1),
}

# Hyperparams to be sent to the Bayesian Optimization (BO) algorithm
def xgb_cl_bo(learning_rate, max_depth, min_child_weight, gamma, colsample_bytree, num_boost_round, reg_lambda, scale_pos_weight, subsample):
    params_xgb = {}
    params_xgb['learning_rate'] = learning_rate
    params_xgb['max_depth'] = round(max_depth)
    params_xgb['min_child_weight'] = min_child_weight
    params_xgb['gamma'] = gamma
    params_xgb['colsample_bytree'] = colsample_bytree
    params_xgb['num_boost_round'] = round(num_boost_round)
    params_xgb['reg_lambda'] = reg_lambda
    params_xgb['scale_pos_weight'] = scale_pos_weight
    params_xgb['subsample'] = subsample
    scores = cross_val_score(XGBClassifier(random_state=123, **params_xgb),
                             X_train, np.ravel(y_train, 'C'), scoring=make_scorer(recall_score), cv=3).mean()
    score = scores.mean()
    return score

# Hyperparameter tuning using BO
start = time.time()

xgb_bo = BayesianOptimization(xgb_cl_bo, params_xgb, random_state=111)
xgb_bo.maximize(init_points=120, n_iter=25)
print('It takes %s minutes' % ((time.time() - start)/60))

# Obtain optimal hyperparameters, based on the BO run
params_xgb = xgb_bo.max['params']
params_xgb['max_depth'] = round(params_xgb['max_depth'])
params_xgb['num_boost_round'] = round(params_xgb['num_boost_round'])

print(params_xgb)

# Can manually enter hyperparams here
params_xgb = {
    'colsample_bytree':  0.619,
    "gamma": 2.106,
    "learning_rate":  0.709,
    "max_depth": 3,
    'min_child_weight': 2.791,
    'num_boost_round': 336,
    "reg_lambda": 4.515,
    "scale_pos_weight": 1.437,
    "subsample": 0.504
}

# Construct final classifier using the BO hyperparameters
final_cl = xgb.XGBClassifier(
    **params_xgb,
)

# Obtain test and train results, after training the final classifier on the train set
_ = final_cl.fit(X_train, y_train)

preds = final_cl.predict(X_test)
predst = final_cl.predict(X_train)

# Score
print("Test acc: ", accuracy_score(y_test, preds))
print("Test rec:", recall_score(y_test, preds))
print("Test pres:", precision_score(y_test, preds))
print()
print("Train acc: ", accuracy_score(y_train, predst))
print("Train rec:", recall_score(y_train, predst))
print("Train pres:", precision_score(y_train, predst))

# Hyperparamter grid for Random Search
param_grid = {
  'learning_rate' : np.arange(0.05, 2, 0.05),
  'max_depth' : np.arange(1, 10, 1),
  'min_child_weight' : np.arange(1, 10, 0.5),
  'gamma': np.arange(0, 3, 0.1),
  'colsample_bytree' : np.arange(0.1, 1, 0.05),
  'num_boost_round': np.arange(100, 500, 50),
  "reg_lambda": np.arange(0.01, 10, 0.05),
  'scale_pos_weight' : np.arange(1, 10, 0.5)
}

# Hyperparameter search using RandomSearch
rand_cv = RandomizedSearchCV(xgb_cl, param_grid, n_iter=210, scoring="accuracy", n_jobs=-1, cv=3, verbose=True)

_ = rand_cv.fit(X_train, y_train)

# Construct and train final classifier based on the RandomSearch's best hyperparams
final_cl = xgb.XGBClassifier(
    **rand_cv.best_params_,
)

_ = final_cl.fit(X_train, y_train)

preds = final_cl.predict(X_test)
predst = final_cl.predict(X_train)

# Score
print("Test acc: ", accuracy_score(y_test, preds))
print("Test rec:", recall_score(y_test, preds))
print("Test pres:", precision_score(y_test, preds))
print()
print("Train acc: ", accuracy_score(y_train, predst))
print("Train rec:", recall_score(y_train, predst))
print("Train pres:", precision_score(y_train, predst))
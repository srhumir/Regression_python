# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:20:40 2016

@author: Hanireza
"""

url = "http://www.qlaym.com/assets/20160816_Qlaym_DS_Assignment_Data.zip"
import urllib.request
urllib.request.urlretrieve(url, "dataset.zip")
import zipfile as zip
import numpy as np
import pandas as pd
file = zip.ZipFile('dataset.zip')
file.namelist()
file.extractall()
data_known = pd.read_csv('dataset_known.csv')

target = data_known["target"]
data_known.drop('target', axis=1, inplace=True)
# checking if some variables are factor even in the form of number
unique_length = np.asarray([data_known[x].unique().size for x in data_known.columns])
factors = unique_length <= 5
#coltypes = data_known.columns.to_series().groupby(data_known.dtypes).groups
# converting variables with less that five unique values to str
data_known[data_known.columns[factors]] = data_known[data_known.columns[factors]].astype(str)
qualind = np.where(data_known.dtypes == 'object')[0]
quanind = np.where(data_known.dtypes == 'float64')[0]

# define dummy variables
data_known_dummy = pd.get_dummies(data_known)

# quantitative variables

known_quan = data_known.iloc[:,quanind]
# seperating qulitative variables
known_qual = data_known.iloc[:,qualind]

import sklearn.preprocessing as pp

# getting correlation between variables
# corMatrix <- cor(known_quan)
 
## imputng nan
imr = pp.Imputer(strategy = "median")
imr.fit(data_known_dummy)
data_known_pre = imr.transform(data_known_dummy)

# divide into train, validation ad test
import sklearn.cross_validation as cv
X_train, X_test, target_train, target_test = \
... cv.train_test_split(data_known_pre, target, test_size = .3, random_state=0)

X_val, X_test, target_val, target_test = \
... cv.train_test_split(X_test, target_test, test_size=.5, random_state=100)

# standardization
std = pp.StandardScaler()
X_train_std = std.fit_transform(X_train)
X_val_std = std.transform(X_val)
X_test_std = std.transform(X_test)

## Feature selection
import sklearn.ensemble as en
feat_lables = data_known_dummy.columns
rf = en.RandomForestRegressor(n_estimators=500,
...                           random_state=0,
...                           n_jobs=-2)
rf.fit(X_train, target_train)
imps = rf.feature_importances_
indices = np.argsort(imps)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_lables[f],imps[indices[f]]))


    print(str(f+1) + ')'+ str(feat_lables[f]) +'     '+ str(imps[indices[f]]))
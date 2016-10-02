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



# quantitative variables
coltypes = data_known.columns.to_series().groupby(data_known.dtypes).groups

known_quan = data_known[data_known.columns[data_known.dtypes == 'float64']]
# seperating qulitative variables
known_qual = data_known[data_known.columns[data_known.dtypes == 'object']]

# dwvwloping model to fill NA
nafill <- preProcess(known_quan, method = "medianImpute")
# filling th NA
known_quan <- predict(nafill, known_quan)

# getting correlation between variables
corMatrix <- cor(known_quan)
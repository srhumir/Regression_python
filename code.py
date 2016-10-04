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
import matplotlib.pyplot as plt
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
## exploratory analysis
cormatrix = known_quan.corr()
# maximum correlation
round(np.max(np.extract(1-np.eye(cormatrix.shape[0]),cormatrix)),2)
# correlation with target
corrs = [known_quan.iloc[:,[i]].assign(target=target).corr().iloc[0,1] \
...     for i in range(0,known_quan.shape[1])]
corrssort = np.argsort(corrs)[::-1]
most_cor = known_quan.columns[corrssort[:4]]
# plot
## quantative variables
plt.figure()
for i in range(0,most_cor.size):
    plt.subplot(221+i)
    plt.scatter(known_quan[most_cor[i]], target)
    plt.title(most_cor[i])
plt.suptitle('Realtionship between quantitative variables and the target')
plt.show()
## qualitative variables
mean_target =[target.groupby(known_qual.iloc[:,i]).mean() 
    for i in range(0,known_qual.shape[1])]
tan_target = [abs(mean_target[i][-1]-mean_target[i][0])/len(mean_target[i])  
    for i in range(0, len(mean_target))]
tansort = np.argsort(tan_target)[::-1]
most_tan = known_qual.columns[tansort[:4]]

a =known_qual[most_tan].assign(target=target)
fig, axes = plt.subplots(figsize=(10,  10), nrows=1, ncols=a.shape[1]-1)
for i in range(0, a.shape[1]-1):
    a.boxplot(column='target', by = most_tan[i], 
              meanline=True, showmeans=True, showcaps=True, 
              showbox=True, showfliers=False, 
              ax=axes[i])
plt.suptitle('Realtionship between qualitative variables and the target')
plt.show()

import sklearn.preprocessing as pp
import math

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
import sklearn.pipeline as pip
rf_pip = pip.Pipeline([('std', pp.StandardScaler()),
                ('rf', en.RandomForestRegressor(n_estimators=500, 
                                                random_state=0, 
                                                n_jobs=-2))])
feat_lables = data_known_dummy.columns

rf = en.RandomForestRegressor(n_estimators=200,
...                           random_state=0,
...                           n_jobs=-2)
rf.fit(X_train_std, target_train)
pred = rf.predict(X_val_std)
imps = rf.feature_importances_
indices = np.argsort(imps)[::-1]
cumsum = np.cumsum(imps[indices])
for f in range(X_train_std.shape[1]):
    print("%2d) %-*s %f %g" % (f + 1, 30,feat_lables[indices[f]],imps[indices[f]], cumsum[f]))

def RMSE(prediction, reference):
    MSE = (prediction - reference).apply(np.square).sum()/prediction.size
    RMSE = math.sqrt(MSE)
    # ME <- sum(abs(prediction-reference))/length(prediction)
    return(RMSE)
# threshod to choose cummulative importance
threshold = .79
thress = np.array(range(73,99))/100
number_vars=[]
train_err = []
val_err=[]
for threshold in thress:
    choose = np.where(cumsum >= threshold)[0][0]
    impvars = indices[:choose]
    rf.fit(X_train_std[:, impvars], target_train)
    number_vars.append(choose+1)
    train_err.append(RMSE(rf.predict(X_train_std[:, impvars]), target_train))
    val_err.append(RMSE(rf.predict(X_val_std[:, impvars]), target_val))
    print("Threshod: " + str(threshold)+ ". " + str(choose+1) + " variables where chosen")
    print("Train RMSE: " + str(train_err[-1]))
    print ("Validation RMSE: " + str(val_err[-1]))

plt.plot(number_vars, train_err,
         color='blue', marker='o',
         label='Training error')
plt.plot(number_vars, val_err,
         color='green', marker='s',
         label='Validation error')
plt.grid()
plt.xlabel('Number of vraibles in the model')
plt.ylabel('Root mean square erro')
plt.legend(loc=7)
plt.show()
#choose the best model
thres = thress[np.where(val_err == np.min(val_err))[0][-1]]
choose = np.where(cumsum >= threshold)[0][0]
impvars = indices[:choose]
rf.fit(X_train_std[:, impvars], target_train)
#residual plot
resids = rf.predict(X_train_std[:, impvars])- target_train
plt.scatter(rf.predict(X_train_std[:, impvars]), resids)


plt.hist(target)
plt.show
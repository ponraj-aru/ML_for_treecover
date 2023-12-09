
# load required libraries here
import os
from osgeo import gdal, ogr, osr, gdalconst, gdal_array
import subprocess, glob
import time
from osgeo.gdalnumeric import *
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import  precision_score,recall_score,average_precision_score,roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import ensemble
import gc
import csv
from gdalconst import GA_ReadOnly
import pickle
from sklearn.metrics import classification_report

train_df = pd.read_csv("/home/ponraj/df_all.csv")
data_X = train_df.drop(["Lat","Lon","tree_cover","lat_lon"],axis=1)
data_y = train_df['tree_cover']
df_all = train_df.drop(["Lat","Lon","lat_lon"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(data_X,data_y,test_size=0.2)

p_test6= {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[1750,2000,2500,3000,3500,4000],'max_depth':[8,9,10,11,12,13],'min_samples_split':[2,4,6,8,10,20,40,60,100], 'min_samples_leaf':[1,3,5,7,9],
          'subsample':[0.8,0.85,0.9,0.95,1]}
tuning = GridSearchCV(estimator =GradientBoostingRegressor(max_features=4 , random_state=10),param_grid = p_test6,n_jobs=4,iid=False, cv=5)
tuning.fit(X_train,y_train)
filename = '/home/ponraj/.sav'
pickle.dump(tuning, open(filename, 'wb'))

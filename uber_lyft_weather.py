#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:33:16 2021

@author: abdul
"""


import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing
from datetime import datetime, timedelta





data1 = pd.read_csv('cab_rides.csv')
data2 = pd.read_csv('weather.csv')

data1 = data1.sort_values('time_stamp')
data2 = data2.sort_values('time_stamp')

#convert date
data1['date_time'] = pd.to_datetime(data1['time_stamp']/1000, unit='s')
data2['date_time'] = pd.to_datetime(data2['time_stamp'], unit='s')


#merge the datasets to refelect same time for a location
data1['merge_date'] = data1.source.astype(str) +" - "+ data1.date_time.dt.date.astype("str") +" - "+ data1.date_time.dt.hour.astype("str")
data2['merge_date'] = data2.location.astype(str) +" - "+ data2.date_time.dt.date.astype("str") +" - "+ data2.date_time.dt.hour.astype("str")


data2.index = data2['merge_date']


merged_df = data1.join(data2,on=['merge_date'],rsuffix ='_w')
merged_df['rain'].fillna(0,inplace=True)


merged_df['day'] = merged_df.date_time.dt.dayofweek


df = pd.merge_asof(data1, data2, on='date_time')

list(df)

df = df.drop('id', axis=1)
df = df.drop('product_id', axis=1)
df = df.drop('rain', axis=1)

df.value_counts()
df['new_date'] = df.time_stamp.apply(str)



df['day'] = df.time_stamp.dt.dayofweek
df['hour'] = df.time_stamp.dt.hour


corr_matrix = data2.corr()
corr_matrix.style.background_gradient(cmap='coolwarm')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
drop_feature2 = [column for column in upper.columns if any(upper[column] > 0.9)]


from datetime import datetime
ts = int('1543203646318')

# if you encounter a "year is out of range" error the timestamp
# may be in milliseconds, try `ts /= 1000` in that case
print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

datetime.datetime.strptime('January 11, 2010', '%B %d, %Y').strftime('%A')

datetime.datetime.strptime('1284101485').strftime('%A')


data3 = df.dropna(thresh=17)


y = data3.pop('EVENT')
X = data3



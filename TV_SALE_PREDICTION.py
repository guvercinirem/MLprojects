#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:46:59 2023

@author: iremguvercin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("/Users/iremguvercin/Downloads/advertising.csv")

def check_df(dataframe):
    print("******HEAD*****")
    print(dataframe.head())
    print("*****TAIL*****")
    print(dataframe.tail())
    print("*****NA*****")
    print(dataframe.isnull().sum())
    print("******TYPE*****")
    print(dataframe.dtypes)
    print("*****INFO******")
    print(dataframe.info())
    print("****SHAPE*****")
    print(dataframe.shape)
    print("*****DESCRIBE****")
    print(dataframe.describe().T)
    
    
check_df(data) 

correlation=data.corr()
   
print(correlation["Sales"].sort_values(ascending=False))    

### Split Data for ML

x=np.array(data.drop(["Sales"],axis=1))
y=np.array(data["Sales"])

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=42)



#MODEL

model=LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))



##output: 0.9059011844150826



#predict feature-sale

features=[["TV","Radio","Newspaper"]]
features=np.array([[243.8,36.9,50.8]])
print(model.predict(features))


#output [21.94867216]



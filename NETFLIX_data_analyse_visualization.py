#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:48:32 2023

@author: iremguvercin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

netflix=pd.read_csv("/Users/iremguvercin/Downloads/netflix_titles.csv")
data=netflix.copy()
data.head()

data.info()
selected_columns=["title","director","cast","country","duration"]
data=data[selected_columns]
data.head()

data.shape

print("In this data set; There are  {} row and {} column" .format(data.shape[0],data.shape[1]))

data.isnull().any()

data["country"].count()
countries=data["country"].value_counts()
countries.head(20)

countries=pd.DataFrame({"Production Number":data["country"].value_counts()})

countries.head(15)

#PRODUCTION NUMBER GRAFFICS

x=countries.index
x=x[0:15]


y=countries["Production Number"]
y=y[0:15]

plt.figure(figsize=(15,8))
plt.bar(x,y)
plt.title("TOP PRODUCTION OF COUNTRIES")
plt.show()



x=countries.index
x=x[0:15]


y=countries["Production Number"]
y=y[0:15]

plt.figure(figsize=(15,8))
color=["g","g","g","g","g","g","g","g","g","g","r","g","g","g","g",]
plt.bar(x,y,color=color)
plt.title("TOP PRODUCTION OF COUNTRIES")
plt.show()




## JUST FOR LOCAL PRODUCTION

local=data.loc[data["country"]=="Turkey"]
local.head()




local=data.loc[data["director"]=="Yılmaz Erdoğan"]
local.head()




data["director"].value_counts()

directors=pd.DataFrame({"Production Number": data["director"].value_counts()})

directors.head(15)

local["director"].value_counts()


director_local=pd.DataFrame({"Production Number":local["director"].value_counts()})




















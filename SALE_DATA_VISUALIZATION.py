#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:12:06 2023

@author: iremguvercin
"""

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("//Users/iremguvercin/Downloads/MOCK_DATA-3.csv")


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Statistics #####################")
    print(dataframe.describe().T)

check_df(data)

#BAR GRAFFIC

x=data["OrderDate"]
y=data["UnitPrice"]
plt.figure(figsize=(10,10))
plt.bar(x,y)
plt.show()

#Can^'t see the dates from graffic
x=range(0,50)
y=data["UnitPrice"]
plt.figure(figsize=(10,10))
plt.bar(x,y,color="purple",width=0.8)
plt.xlabel("DATE")
plt.ylabel("UNIT-PRICE")
plt.title("SALES")
plt.xticks(range(0,50,5),range(0,50,5))
plt.plot(x,y,color="blue")
plt.legend(labels=["Satışlar"])
plt.show()

sales=data.groupby("Item")
sales=sales.sum()
sales=sales.plot(kind="bar")
sales.set_ylabel("Unit & Unit Price")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:30:05 2023

@author: iremguvercin
"""


# import necessary libraries

import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df=pd.read_csv("/Users/iremguvercin/Downloads/Credit card data/CC GENERAL.csv")

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(7))
    print("##################### Tail #####################")
    print(dataframe.tail(7))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe().T)


check_df(df)

df=df.dropna()

cluster_df=df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

for i in cluster_df.columns:
    MinMaxScaler(i)
    
kmeans=KMeans(n_clusters=5) 
clusters=kmeans.fit_predict(cluster_df)
df["SEGMENT"]=clusters



df["SEGMENT"]=df["SEGMENT"].map({0:"Cluster1",1:"Cluster2",
                                 2:"Cluster3",3:"Cluster4",4:"Cluster5"})
df["SEGMENT"].head(20)




import matplotlib.pyplot as plt
import seaborn as sns  


sns.scatterplot(x='SEGMENT', y='CREDIT_LIMIT', hue='SEGMENT', data=df, palette='viridis')
plt.title('Credit Card Clusters')
plt.xlabel('SEGMENT')
plt.ylabel('CREDIT_LIMIT')
plt.show()


sns.scatterplot(x='SEGMENT', y='PURCHASES', hue='SEGMENT', data=df, palette='viridis')
plt.title('Credit Card Clusters')
plt.xlabel('SEGMENT')
plt.ylabel('PURCHASES')
plt.show()




sns.scatterplot(x='SEGMENT', y='BALANCE', hue='SEGMENT', data=df, palette='viridis')
plt.title('Credit Card Clusters')
plt.xlabel('SEGMENT')
plt.ylabel('BALANCE')
plt.show()












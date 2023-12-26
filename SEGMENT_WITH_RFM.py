#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 23:50:05 2023

@author: iremguvercin
"""

import pandas as pd
import numpy as np

flo=pd.read_csv("/Users/iremguvercin/Downloads/FLO_RFM_Analizi/flo_data_20K.csv")
df=flo.copy()


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

check_df(df)

df.columns
df.info()
df.shape
df.order_channel.value_counts()
df.store_type.value_counts()
df.last_order_channel.value_counts()
df.order_num_total_ever_offline.value_counts()


##NEW FEATURES

df["order_num_total"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]

##DATE TYPE CORRECTING FROM OBJECT TO DATE

import datetime as dt

date_columns=df.columns[df.columns.str.contains("date")]
df[date_columns]=df[date_columns].apply(pd.to_datetime)
df.info()
df.last_order_date.max()


"""Out[15]: Timestamp('2021-05-30 00:00:00')"""




df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})



import seaborn as sns
import matplotlib.pyplot as plt


sns.lineplot(x=df.order_num_total, y=df.customer_value_total,data=df)
plt.show()
plt.figure(figsize=(10,6))
sns.set(rc={"axes.facecolor":"white",
            "axes.grid":True,
            "xtick.labelsize":12,
            "ytick.labelsize":12})



#TOP 10 CUSTOMER-VALUE

df.sort_values("customer_value_total",ascending=False).head(10)

#TOP 10 ORDER AMOUNT

df.sort_values("order_num_total",ascending=False).head(10)


#MAKE PRE EDA IN A FUNCTION

def data_prep(data):
    df["order_num_total"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
    df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]

    date_columns=df.columns[df.columns.str.contains("date")]
    df[date_columns]=df[date_columns].apply(pd.to_datetime)
    return data

data_prep(df)



#SET RFM METRICS

df["last_order_date"].max() #2021-05-30
analysis_date=dt.datetime(2021,6,1)

# MAKE A NEW DATAFRAME WITH COSTUMER_ID,RECENCY,FREQUENCY,MONETARY

import datetime as dt

rfm=pd.DataFrame()
rfm["costomer_id"]=df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days

rfm["frequency"]=df["order_num_total"]
rfm["monetary"]=df["customer_value_total"]


rfm.head(10)


##CALCULATING THE RFM SCORES
##RECENCY,MONETARY,FREQUENCY DIVIDE IN 1-5 WITH QCUT METHOT
##TRANSFORMING TO SCORES like recency_score, monetary_score, frequency_score


rfm["recency_score"]=pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["monetary_score"]=pd.qcut(rfm["monetary"],5,labels=[1,2,3,4,5])
rfm["frequency_score"]=pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])




rfm.head()


rfm["RF_SCORE"]=rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
rfm["RFM_SCORE"]=rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str)


rfm.head()


#GIVING -NAME TO RFM SCORES WTIH SEG-MAP

#RFM SKORLARININ SEGMENT OLARAK TANIMLANMASI- seg_map yardımı ile.

seg_map={ r'[1-2][1-2]':'hibernating',
          r'[1-2][3-4]':'at_Risk',
          r'[1-2]5':'cant_loose',
          r'3[1-2]':'about_to_sleep',
          r'33':'need_attention',
          r'[3-4][4-5]':'loyal_customers',
          r'41':'promising',
          r'51':'new_customers',
          r'[4-5][2-3]':'potential_loyalists',
          r'5[4-5]':'champions'}




rfm["segment"]=rfm["RF_SCORE"].replace(seg_map,regex=True)


##EXAMINE SEGMENT' RECENCY,MONETARY,FREQUENCY MEANS



rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])



### REVIEW CASE WITH FRM ANALYSE
## Flo is adding a new women's shoe brand to its portfolio.
# Prices of the included brand are above the general price average.

# For this reason, we want to communicate specifically with customers who will be interested in the brand.
# These people were planned to be loyal, women's category shoppers who shop for an average of 250 TL and above.
# Save the id numbers of the customers in the csv file as new_brand_target_customer_id.csv.



target_segments_customer_ids= rfm[rfm["segment"].isin(["champions","loyal_customers"])]["costomer_id"]

cust_ids=df[(df["master_id"].isin(target_segments_customer_ids))&(df["customer_value_total"]/df["order_num_total"]>250)&
(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

cust_ids.to_csv("NEW_CUST_TARGET_ID",index=False)



cust_ids.shape


###Nearly 40% discount is planned for men's and children's products. 
##These discounts are related to categories who have been good in the past but haven't shopped in a long time
#and new customers are wanted to be specifically targeted.
#Save the ids of the customers in the appropriate profile into the csv file
#Save it as #discount_target_customer_ids.csv.


target_segments_customer_ids= rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["costomer_id"]

cust_ids=df[(df["master_id"].isin(target_segments_customer_ids))&
(df["interested_in_categories_12"].str.contains("ERKEK","ÇOCUK"))]["master_id"]

cust_ids.to_csv("SALE_CUST_TARGET_ID",index=False)

cust_ids.shape










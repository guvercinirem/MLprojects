#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:10:02 2023

@author: iremguvercin
"""

#import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns






pd.options.display.float_format="{:,.0f}".format 

import warnings
warnings.filterwarnings("ignore")

#load dataset

orders=pd.read_csv("/Users/iremguvercin/Downloads/orders.csv")

df=orders.copy()
df.head()
df.info()

#DateTime_CartFinalize  200000 non-null  object / needs change to datetime

df["purchase_date"]=pd.to_datetime(df["DateTime_CartFinalize"],format='%Y-%m-%d')

df.drop('DateTime_CartFinalize',axis=1,inplace=True)

df.head()
df.info()
df.shape

#ID_Items            200000 non-null  int64 /needs change to text

df["ID_Items"]=df["ID_Items"].astype("str")

df[['Amount_Gross_Order','Quantity_item']].describe()

#EXPLORATORY DATA ANALYSIS

df['purchase_date'].max()

"""Out[6]: Timestamp('2018-12-11 04:47:23')"""



#RFM for Customer Segmentation

#Based on R_ecency, F_requency and M_onetary identify the customer groups.

# Doing RFM with K_Means Clustering

def outlier_drop(data,column):
    Q1=np.percentile(data[column],25,interpolation="midpoint")
    Q3=np.percentile(data[column],75,interpolation="midpoint")
    
    IQR=Q3-Q1 
    upper_limit=Q3 + IQR*1.5
    lower_limit=Q1 - IQR*1.5
    
    return data[(data[column]<upper_limit) & (data[column]>lower_limit)]

outlier_drop(df, 'Amount_Gross_Order')


today_date=df["purchase_date"].max()

cv=df.groupby('ID_Customer').agg(
    frequency=('ID_Order',pd.Series.nunique),
    recency=('purchase_date',lambda x: (today_date - x.max()).days+1),
    monetary=('Amount_Gross_Order',np.sum),
    tenure=('purchase_date',lambda x: (today_date - x.min()).days+1))
                               
cv.info()
cv=outlier_drop(cv, 'monetary')
cv=cv[cv['monetary']!=0]
                                 

sns.scatterplot(data=cv,x="recency",y="tenure")
sns.histplot(data=cv,x="monetary")
sns.histplot(data=cv,x="recency")

customers=cv.copy()

customer_normalized=customers.iloc[:,0:4].copy() # customers veri setindeki frequency ve monetary değişkenlerini içeren sütunlarının olduğu yeni bir dataframe
customer_normalized.head()  
customers.head()                   
sns.histplot(data=customer_normalized,x="monetary")

sns.histplot(data=customer_normalized,x="recency")




from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler and apply it to the data
scaler = StandardScaler()
customer_normalized = scaler.fit_transform(customers)



"""Elbow yöntemi, k-ortalama (k-means) kümeleme algoritması için optimal 
küme sayısını belirlemek için kullanılan bir yöntemdir. Küme sayısı 
arttıkça küme içi varyans azalır, ancak küme sayısı çok fazla 
olduğunda aşırı öğrenme (overfitting) problemi ortaya çıkabilir. 
Elbow yöntemi, farklı küme sayıları için küme içi varyansı hesaplar 
ve bu değerlerin grafiğini çizer. Grafiğin şekline bakarak, 
küme sayısı için en iyi seçimin yapılması için küme sayısının 
arttırılması ile varyans azalma hızının düştüğü nokta seçilir. 
Bu nokta, "dirsek" (elbow) olarak adlandırılır ve en iyi küme 
sayısı olarak belirlenir."""





from sklearn.cluster import KMeans

sse = {}
# Fit KMeans and calculate SSE for each k between 1 and 10
for k in range(1, 11):
  
    # Initialize KMeans with k clusters and fit it 
    kmeans = KMeans(n_clusters = k, random_state = 1)
    kmeans.fit(customer_normalized)
    
    # Assign sum of squared distances to k element of the sse dictionary
    sse[k] = kmeans.inertia_

# Add the plot title, x and y axis labels
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')

# Plot SSE values for each k stored as keys in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()



                
kmeans = KMeans(n_clusters=2, random_state=1)
kmeans.fit(customer_normalized)
cluster_labels = kmeans.labels_


# Assign cluster labels
customers = customers.assign(Cluster=cluster_labels)
#customers_df['Cluster'] = cluster_labels

customers.head()

# Group by cluster
grouped = customers.groupby(['Cluster']).agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'tenure': ['mean', 'count']
}).round(1)    
grouped.head()

"""ecency frequency monetary tenure       
           mean      mean     mean   mean  count
Cluster                                         
0           894         1  758,764    996  49268
1           275         1  673,557    318  84634"""


# Calculate average RFMT values and segment sizes for each cluster
                
                                 
                                
                                 
                                 
figure, axs = plt.subplots(1,3, figsize=(15,6))

fig1 = sns.boxplot(data=customers, y='monetary', x='Cluster', ax=axs[0])
fig2 = sns.boxplot(data=customers, y='recency', x='Cluster', ax=axs[1])
fig3 = sns.boxplot(data=customers, y='tenure', x='Cluster', ax=axs[2])

ylabels = ['{:,.1f}'.format(x) for x in fig1.get_yticks()/1000000]
fig1.set(xlabel=None, ylabel='Monetary (in millions IRR)')


fig1.set_yticklabels(ylabels)

fig2.set(xlabel=None, ylabel = 'Recency (in days)')
fig3.set(xlabel=None, ylabel = 'Tenure (in days)')

figure.suptitle('Monetary, Recency and Tenure of every Cluster')
plt.show()                                 
                                 
                                 
#Cluster 0: GOOD CUSTOMERS who prefers digikala more than cluster 1
#Cluster 1: 
                            
                                 
                                 
customers.head()    


df.head()    

merge_df=pd.merge(customers,df,on="ID_Customer")                        
                                 
                                 
                                 
merge_df.head()                                 
                 

pd.set_option("display.max_columns",500)          
                                 
pd.set_option("display.max_rows",500)                                
                                 

merge_df=merge_df[["ID_Customer","Amount_Gross_Order","purchase_date","Cluster","recency","frequency","monetary","tenure"]]
                                 
                                  
merge_df.head()
merge_df.info()  
merge_df.isnull().sum()

merge_df["purchase_date"].max()
#Out[24]: Timestamp('2018-12-11 04:47:23')


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve     
import pandas as pd
from sklearn.model_selection import GridSearchCV
merge_df_train = merge_df[(merge_df['purchase_date'] <= '2018-06-10') & (merge_df["purchase_date"]>='2018-01-07')]
merge_df_test = merge_df[(merge_df["purchase_date"] < '2018-12-11') & (merge_df["purchase_date"] > '2018-06-10') ]                          

# Tarihleri sayısal değerlere dönüştür
merge_df_train['purchase_date'] = pd.to_datetime(merge_df_train['purchase_date']).astype(int)
merge_df_test['purchase_date'] = pd.to_datetime(merge_df_test['purchase_date']).astype(int)

y = merge_df_test["Cluster"]
X = merge_df_train.drop(["Cluster"], axis=1)

print(X.shape)
print(y.shape)

y = y[:X.shape[0]]

#Logistik regression'u denersem
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
parametreler = {'penalty': ['l1', 'l2'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, parametreler, cv=5)
grid_search.fit(X_train, y_train)
print("En İyi Parametreler: ", grid_search.best_params_)
print("En İyi Skor: ", grid_search.best_score_)
en_iyi_model = grid_search.best_estimator_
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Test seti doğruluğu", accuracy)

#CLUSTER'I BINARY YE DÜŞÜRÜNCE Test seti doğruluğu 0.047

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
report = classification_report(y_test, y_pred)
print(report)


#2 SEGMENT İLE DOĞRULUK 
"""Test seti doğruluğu 0.04784763084546299
[[ 309    0]
 [6149    0]]
              precision    recall  f1-score   support

           0       0.05      1.00      0.09       309
           1       0.00      0.00      0.00      6149

    accuracy                           0.05      6458
   macro avg       0.02      0.50      0.05      6458
weighted avg       0.00      0.05      0.00      6458"""

#5 SEGMENT İLE DOĞRULUK
"""Test seti doğruluğu 0.016
[[ 109    0    0    0]
 [ 784    0    0    0]
 [ 946    0    0    0]
 [4619    0    0    0]]
              precision    recall  f1-score   support

           0       0.02      1.00      0.03       109
           1       0.00      0.00      0.00       784
           2       0.00      0.00      0.00       946
           3       0.00      0.00      0.00      4619

    accuracy                           0.02      6458
   macro avg       0.00      0.25      0.01      6458
weighted avg       0.00      0.02      0.00      6458"""

#KARAR AĞACI ALGORİTMASI DENERSEM


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Test seti doğruluğu", accuracy)

#Test seti doğruluğu 0.30659646949519975

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))

#2 SEGMENT 
"""Test seti doğruluğu 0.9143697739238155
[[  34  275]
 [ 278 5871]]
              precision    recall  f1-score   support

           0       0.11      0.11      0.11       309
           1       0.96      0.95      0.96      6149

    accuracy                           0.91      6458
   macro avg       0.53      0.53      0.53      6458
weighted avg       0.91      0.91      0.91      6458"""

#5 SEGMENT

"""[Test seti doğruluğu 0.5425828429854445
[[   1   10   27   71]
 [  19   98  113  554]
 [  23   99  189  635]
 [  82  630  691 3216]]
              precision    recall  f1-score   support

           0       0.01      0.01      0.01       109
           1       0.12      0.12      0.12       784
           2       0.19      0.20      0.19       946
           3       0.72      0.70      0.71      4619

    accuracy                           0.54      6458
   macro avg       0.26      0.26      0.26      6458
weighted avg       0.56      0.54      0.55      6458"""






#RANDOM FOREST 


from sklearn.ensemble import RandomForestClassifier

# Random Forest sınıflandırıcısını çağırma
model = RandomForestClassifier()

# Modeli eğitme
model.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Test seti doğruluğu
accuracy = model.score(X_test, y_test)
print("Test seti doğruluğu", accuracy)



# Sınıflandırma raporu
print(classification_report(y_test, y_pred))

#SEGMENT 2

"""Test seti doğruluğu 0.9516878290492412
              precision    recall  f1-score   support

           0       0.29      0.01      0.01       309
           1       0.95      1.00      0.98      6149

    accuracy                           0.95      6458
   macro avg       0.62      0.50      0.49      6458
weighted avg       0.92      0.95      0.93      6458"""

#SEGMENT 5

"""Test seti doğruluğu 0.40430473830907404
              precision    recall  f1-score   support

           precision    recall  f1-score   support

           0       0.46      0.77      0.58      2958
           1       0.17      0.06      0.08       905
           2       0.08      0.02      0.03       397
           3       0.17      0.03      0.06       615
           4       0.26      0.17      0.20      1583

    accuracy                           0.40      6458
   macro avg       0.23      0.21      0.19      6458
weighted avg       0.32      0.40      0.33      6458"""
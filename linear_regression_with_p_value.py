#!/usr/bin/env python
# coding: utf-8

# In[3]:


#libraries : numpy, pandas, matplotlib, seaborn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#linear regression from sklearn

from sklearn.linear_model import LinearRegression

DF=pd.read_csv("MOCK_DATA-4.csv")
DF.head()


# In[9]:


DF.shape
DF.columns=["sale","size","distance"]


# In[10]:


DF.describe().T


# In[12]:


### TARGET & INDEPENDENT VARIES ### REGRESYON


x=DF[["size","distance"]]
y=DF["sale"]

##SCATTER GRAFFIC

plt.scatter(x["size"],y)
plt.xlabel("size", fontsize=15)
plt.ylabel("sale", fontsize=15)
plt.show()


plt.scatter(x["distance"],y)
plt.xlabel("distance", fontsize=15)
plt.ylabel("sale", fontsize=15)
plt.show()



# In[18]:


#####REGRESSION

reg=LinearRegression()
reg.fit(x,y)

reg.intercept_
reg.coef_


# In[19]:


#r kare skor- 1 e ne kadar yakınsqa o kadar iyi

reg.score(x,y)


# In[20]:


# Düzeltilmiş r kare
r2=reg.score(x,y)
n=x.shape[0]
p=x.shape[1]
düzeltilmiş_r2=1-(1-r2)*(n-1)/(n-p-1)
düzeltilmiş_r2


# In[22]:


#PREDICTION

reg.predict([[150,45]])


# In[28]:


#NEW PREDICT DATAFRAME
new_data.rename(columns={"boyut":"size"},inplace=True)


# In[29]:


predictions=reg.predict(new_data).round(1)


# In[30]:


new_data["Predict Sale"]=reg.predict(new_data)


# In[31]:


new_data.head()


# In[ ]:


## CALCULATING P_VALUE

from sklearn.feature_selection import f_regression
f_regression(x,y)
#p_values




# he p-value is a number that describes how likely it is that events occurred by chance (i.e., the null hypothesis is true).
# 
# The level of statistical significance is expressed as a p-value between 0 and 1. The smaller the p-value, the stronger the evidence that you should reject the null hypothesis.
# 
# The threshold value (also referred to as the critical value or alpha value) depends on the study area. Some fields prefer thresholds of 0.01 or even 0.001. However, the most common threshold is p < 0.05.

# In[32]:


from sklearn.feature_selection import f_regression
f_regression(x,y)
p_values=f_regression(x,y)[1]
p_values


# In[33]:


p_values.round(3)


# In[34]:


reg_summary=pd.DataFrame(data=x.columns.values,columns=["Features"])
reg_summary["Weights"]=reg.coef_
reg_summary["p_values"]=p_values.round(3)
reg_summary


# In[ ]:





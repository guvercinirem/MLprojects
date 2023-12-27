#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:40:14 2023

@author: iremguvercin

"""



#################
# Telco Customer Churn Feature Engineering
#################

# Problem: It is desired to develop a machine learning model that can predict customers who will leave the company.
# You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

# Customer churn data from Telco, which provides home phone and Internet services to 7,043 customers in California in the third quarter
# contains information about a fictitious telecom company. It includes which customers left, stayed, or signed up for their service.

#21 Variable 7043 Observations

# CustomerId : Customer ID
# Gender
# SeniorCitizen : Whether the customer is elderly (1, 0)
# Partner: Whether the customer has a partner (Yes, No)? married or not
# Dependents: Whether the customer has dependents (Yes, No) (Child, mother, father, grandmother)
# tenure: Number of months the customer stays with the company
# PhoneService: Whether the customer has phone service (Yes, No)
# MultipleLines: Whether the customer has more than one line (Yes, No, No phone service)
# InternetService: Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, no Internet service)
# OnlineBackup: Whether the customer has an online backup (Yes, No, no Internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, no Internet service)
# TechSupport: Whether the customer receives technical support (Yes, No, no Internet service)
# StreamingTV : Whether the customer has streaming TV (Yes, No, No Internet service) Indicates whether the customer uses Internet service to stream television programs from a third-party provider
# StreamingMovies : Whether the customer is streaming movies (Yes, No, No Internet service) Indicates whether the customer is using Internet service to stream movies from a third-party provider
# Contract: Customer's contract period (Month to month, One year, Two years)
# PaperlessBilling: Whether the customer has a paperless bill (Yes, No)
# PaymentMethod: Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: Amount collected from the customer monthly
# TotalCharges: Total amount collected from the customer
# Churn: Whether the customer used (Yes or No) - Customers who left in the last month or quarter


# Each row represents a unique customer.
# Variables contain information about customer service, account and demographic data.
# Services that customers sign up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information – how long they have been a customer, contract, payment method, paperless billing, monthly fees and total charges
# Demographic information about customers – gender, age range and whether they have partners and dependents

# 1: EXPLORATORY DATA ANALYSIS
# Step 1: Examine the general picture.
# Step 2: Capture numerical and categorical variables.
# Step 3: Analyze numerical and categorical variables.
# Step 4: Perform target variable analysis. (Average of target variable according to categorical variables, average of numerical variables according to target variable)
# Step 5: Perform an outlier observation analysis.
# Step 6: Perform missing observation analysis.
# Step 7: Perform correlation analysis.

# 2: FEATURE ENGINEERING
# Step 1: Take the necessary action for missing and outlier values.
# You can apply operations.
# Step 2: Create new variables.
# Step 3: Perform the encoding operations.
# Step 4: Standardize numerical variables.
# Step 5: Create the model.

# IMPORT LIBRARIES AND FUNCTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
#from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


telco_churn = pd.read_csv(
    "/Users/iremguvercin/Downloads/TelcoChurn/Telco-Customer-Churn.csv")

df = telco_churn.copy()


# 1: EXPLORATORY DATA ANALYSIS
# Step 1: Examine the general picture.


def check_df(dataframe):
    print("SHAPE")
    print(dataframe.shape)
    print("INFO")
    print(dataframe.info())
    print("TYPE")
    print(dataframe.dtypes)
    print("NULL")
    print(dataframe.isnull().sum())
    print("STATISTICS")
    print(dataframe.describe().T)


check_df(df)


df.head()


# Totalcharges seem object type, should be numerical.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# Target feature "churn" needs to change to 1 and 0 from yes and no.

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)


# 1: EXPLORATORY DATA ANALYSIS
# Step 2: Capture numerical and categorical variables.


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included.

    parameters
    ------
        dataframe: dataframe
                Dataframe from which variable names are to be taken
        cat_th: int, optional
                Class threshold value for variables that are numeric but categorical
        car_th: int, optional
                class threshold for categorical but cardinal variables

    returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                List of cardinal variables with categorical view

    examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique(
    ) < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() >
                   car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols+num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols +num_cols+cat_but_car = number of features
    #num_but_cat is already in cat_cols
    # because of those: we should choose all features with cat_cols + num_cols + cat_but_car
    # num_but_cat is just given for reporting
    return cat_cols, cat_but_car, num_cols


cat_cols, cat_but_car, num_cols = grab_col_names(df)



"""output  :
    Observations: 7043
    Variables: 21
    cat_cols: 17
    num_cols: 3
    cat_but_car: 1
    num_but_cat: 2
    """
    
    
    
# 1: EXPLORATORY DATA ANALYSIS
    # Step 3: Analyze numerical and categorical variables.    
    
        #CATEGORICAL ANALYZE FIRST
        



def cat_analyze(dataframe,col_name,plot=True):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("******")
    print("++++++")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()
        
        
for col in cat_cols:
    cat_analyze(df, col)        
        
        
          
     #NUMERICAL ANALYZE TIME
     
def num_analyze(dataframe,numerical_col,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
         dataframe[numerical_col].hist(bins=20)
         plt.xlabel(numerical_col)
         plt.title(numerical_col)
         plt.show()
         
for col in num_cols:
    num_analyze(df, col,plot=False)         
    
    
    
# When we look at Tenure, we see that there are a lot of 1-month customers, followed by 70-month customers.
# It may have occurred due to different contracts. Let's look at the tenure of people with monthly contracts and the tenure of people with 2-year contracts.    

df[df["Contract"]=="Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month to Month")
plt.show()

df[df["Contract"]=="Two year"]["tenure"].hist(bins=20)
plt.xlabel("Tenure")
plt.title("Two year")
plt.show()

##Looking at  MonthyChargers, customers with monthly contracts may have higher average monthly payments.


df[df["Contract"]=="Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("Monthly Charges")
plt.title("Month to month")
plt.show()


df[df["Contract"]=="Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("Monthly Charges")
plt.title("Two year")
plt.show()






# 1: EXPLORATORY DATA ANALYSIS

    # Step 4: Perform target variable analysis. (Average of target variable according to categorical variables, average of numerical variables according to target variable)


# ANALYSIS OF NUMERICAL VARIABLES BY TARGET



def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n")
    
for col in num_cols:

    target_summary_with_num(df, "Churn", col)    
          




# ANALYSIS OF CATEGORICAL VARIABLES BY TARGET



def target_summary_with_cat(dataframe,target,categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"Target Mean":dataframe.groupby(categorical_col)[target].mean(),
                        "Count":dataframe[categorical_col].value_counts(),
                        "Ratio":100 * dataframe[categorical_col].value_counts()/len(dataframe)}),end="\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)




###CORRELATION

df[num_cols].corr()

###CORRELATION MATRIS


f,ax=plt.subplots(figsize=[18,13]) #This line creates a matplotlib figure and its axis. The figsize parameter determines the dimensions of the drawing.

sns.heatmap(df[num_cols].corr(),annot=True,fmt=".2f",ax=ax, cmap="magma")#This line plots the correlation matrix using the heatmap function of the Seaborn library. The expression df[num_cols].corr() creates the correlation matrix of the numeric columns in the data frame. The annot=True parameter allows numeric values ​​to be added to the cells of the matrix. The fmt=".2f" parameter displays numbers in decimal format. The ax=ax parameter specifies on which axis the drawing will be made. The cmap="magma" parameter determines the color map.

ax.set_title("Correlation Matrix",fontsize=20)
plt.show()






##################################
# 2: FEATURE ENGINEERING
##################################

##################################
# MISSING VALUE ANALYSIS
##################################


df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)  # totalcharge can be filled with monthly payments (better give it a try) or 11 variables can be dropped

df.isnull().sum()


###BASE MODEL SETUP


dff=df.copy()
cat_cols=[col for col in cat_cols if col not in ["Churn"] ]



def one_hot_encoder(dataframe,categorical_cols,drop_first=False):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe


dff=one_hot_encoder(dff, cat_cols,drop_first=True)



y=dff["Churn"]
x=dff.drop(["Churn","customerID"],axis=1)



models=[("LR",LogisticRegression(random_state=12345)),
        ("KNN",KNeighborsClassifier()),
        ("CART",DecisionTreeClassifier(random_state=12345)),
        ("RF",RandomForestClassifier(random_state=12345)),
        ("SVM",SVC(gamma="auto",random_state=12345)),
        ("XGB",XGBClassifier(random_state=12345)),
        ("CatBoost",CatBoostClassifier(verbose=False,random_state=12345))]



for name,model in models:
    cv_results=cross_validate(model,x,y,cv=10, scoring= ["accuracy","f1","roc_auc","precision","recall"])





print(f"###***{name}***###")
print(f"Accuracy:{round(cv_results['test_accuracy'].mean(),4)}")
print(f"Auc:{round(cv_results['test_roc_auc'].mean(),4)}")

print(f"Recall:{round(cv_results['test_recall'].mean(),4)}")

print(f"Precision:{round(cv_results['test_precision'].mean(),4)}")

print(f"F1:{round(cv_results['test_f1'].mean(),4)}")


"""###***CatBoost***###
Accuracy:0.8001
Auc:0.8413
Recall:0.5131
Precision:0.6595
F1:0.5767"""



################################################
# Random Forests
################################################




rf_model=RandomForestClassifier(random_state=17)
rf_params={"max_depth":[5,8,None],
           "max_features":[3,5,7,"auto"],
           "min_samples_split":[2,5,8,15,20],
           "n_estimators":[100,200,500]}




rf_best_grid=GridSearchCV(rf_model, rf_params,cv=5,n_jobs=-1,verbose=True).fit(x,y)


"""Fitting 5 folds for each of 180 candidates, totalling 900 fits"""


rf_best_grid.best_params_


rf_best_grid.best_score_

rf_final=rf_model.set_params(**rf_best_grid.best_params_,random_state=17).fit(x,y)



cv_results = cross_validate(rf_final, x, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

####\\\\\\\Out[171]: 0.8448109553300608

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(x, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(x, y)

cv_results = cross_validate(xgboost_final, x, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()





"""################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
"""



################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(x, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(x, y)

cv_results = cross_validate(catboost_final, x, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()




################################################
# Feature Importance
################################################

def plot_importance(model,features,num=len(x),save=False):
    feature_imp=pd.DataFrame({"Value":model.feature_importances_,
                              "Feature":features.columns})

    plt.figure(figsize=(10,10))
    sns.set(font_scale=1.5)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')




plot_importance(rf_final, x)
plot_importance(xgboost_final, x)
#plot_importance(lgbm_final, x)
plot_importance(catboost_final, x)










































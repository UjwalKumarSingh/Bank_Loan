#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Bank Data analysis for predicting default customer


# In[ ]:


import os


# In[ ]:


def current_path(): 
    print("Current working directory before") 
    print(os.getcwd()) 
    print() 


# In[ ]:

os.chdir('F:\Study\DATA_SCIENCE\MAIN_PROJECT\DATA_SCIENCE\spyer_run')
current_path()


# In[ ]:


#Basic libraries
import pandas as pd
import numpy as np
from   numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import datetime
import math
import pickle
from datetime import date
from scipy import stats


# In[ ]:


#Fetaure Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[ ]:


#Imbalance Dataset
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# In[ ]:


#Model Evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, f1_score,auc,roc_curve,roc_auc_score, precision_recall_curve
import scikitplot as skplt


# In[ ]:


#Modelling Algoritm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[ ]:


bank_data = pd.read_csv('F:\\Study\\DATA_SCIENCE\\MAIN_PROJECT\\DATA_SCIENCE\\bank_final.csv').drop_duplicates()
bank_data.head()


# In[ ]:


bank_data.dtypes


# In[ ]:


bank_data.isnull().sum()


# In[ ]:


## Changing the date types
date_col = ['ApprovalDate', 'ChgOffDate','DisbursementDate']
bank_data[date_col] = pd.to_datetime(bank_data[date_col].stack(),format='%d-%b-%y').unstack()


# In[ ]:


bank_data.head()


# In[ ]:


bank_data.dtypes


# In[ ]:


## Removing the dolloar sign from amount daa columns
col = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
bank_data[col] = bank_data[col].replace('[\$,]', '', regex=True).astype(float)


# In[ ]:


bank_data[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].head()


# In[ ]:


## converting MIS status data as 0 and 1.
bank_data['MIS_Status'].unique()


# In[ ]:


bank_data['MIS_Status'] = bank_data['MIS_Status'].replace({'P I F': 0, 'CHGOFF':1})
bank_data.MIS_Status.value_counts()
# Mis null value check later


# In[ ]:


##LowDoc column treatment
bank_data['LowDoc'] = bank_data['LowDoc'].replace({'[C,1]':np.nan})
bank_data['LowDoc'] = bank_data['LowDoc'].replace({'N': 0, 'Y':1})
bank_data['LowDoc'] = np.where((bank_data['LowDoc'] != 0) & (bank_data['LowDoc'] != 1), np.nan, bank_data.LowDoc)
bank_data.LowDoc.value_counts()


# In[ ]:


bank_data['LowDoc'].unique()


# In[ ]:


# RevLineCr column treatment
bank_data['RevLineCr'] = bank_data['RevLineCr'].replace({'N': 0, 'Y':1, })
bank_data['RevLineCr'] = bank_data['RevLineCr'].replace({'0': 0, '1':1, })
bank_data['RevLineCr'] = np.where((bank_data['RevLineCr'] != 0) & (bank_data['RevLineCr'] != 1), np.nan, bank_data.RevLineCr)
bank_data.RevLineCr.value_counts()


# In[ ]:


# Replacing 0 as null for New Exist
bank_data['NewExist'] = bank_data['NewExist'].replace({1.0: 0, 2.0:1, 0:np.nan}).fillna(0).astype(int)
bank_data.NewExist.value_counts()


# In[ ]:


bank_data['NewExist'].unique()


# In[ ]:


##*************Create flag column IsFranchise based on FranchiseCode column 
bank_data.loc[(bank_data['FranchiseCode'] <= 1), 'IsFranchise'] = 0
bank_data.loc[(bank_data['FranchiseCode'] > 1), 'IsFranchise'] = 1

# Format dtypes where necessary
bank_data = bank_data.astype({'IsFranchise': 'int64'})


# In[ ]:


#CreateJob column treatment
bank_data['CreateJob'] = np.where((bank_data.CreateJob > 0 ),1,bank_data.CreateJob)
bank_data.CreateJob.value_counts()


# In[ ]:


bank_data['CreateJob'].unique()


# In[ ]:


#RetainJob column treatment
bank_data['RetainedJob'] = np.where((bank_data.RetainedJob > 0 ),1,bank_data.RetainedJob)
bank_data.RetainedJob.value_counts()


# In[ ]:


# Create DisbursementFY field for time selection criteria later
bank_data['DisbursementFY'] = bank_data['DisbursementDate'].map(lambda x: x.year)


# In[ ]:


# Field for loans active during the Great Recession (2007-2009)
bank_data['GreatRecession'] = np.where(((2007 <= bank_data['DisbursementFY']) & (bank_data['DisbursementFY'] <= 2009)) | 
                                     ((bank_data['DisbursementFY'] < 2007) & (bank_data['DisbursementFY'] + (bank_data['Term']/12) >= 2007)), 1, 0)


# In[ ]:


bank_data['GreatRecession'].unique()


# In[ ]:


bank_data.shape


# In[ ]:


# Selects only the first two numbers of the CCSC code as business Type
bank_data['BusinessType'] = bank_data['CCSC'].astype('str').apply(lambda x: x[:2])


# In[ ]:


bank_data['BusinessType'].unique()


# In[ ]:


bank_data['BusinessType'] = bank_data['BusinessType'].astype('int64')


# In[ ]:


bank_data.isnull().sum()


# In[ ]:


## NUll value treatment RevlIneCrc 
bank_data['RevLineCr'] = np.where((bank_data['RevLineCr'] == np.nan) & (bank_data['MIS_Status']=='CHGOFF'),1,bank_data.RevLineCr)
bank_data['RevLineCr'] = np.where((bank_data['RevLineCr'] == np.nan) & (bank_data['MIS_Status']=='P I F'),0,bank_data.RevLineCr)

bank_data = bank_data[(bank_data['RevLineCr'] == 0) | (bank_data['RevLineCr'] == 1)]
bank_data.RevLineCr.value_counts()

bank_data['RevLineCr'].unique()


# In[ ]:


## NUll value treatment LowDoc
bank_data['LowDoc'] = np.where((bank_data['LowDoc'] == np.nan) & (bank_data['DisbursementGross'] < 150000),1,bank_data.LowDoc)
bank_data['LowDoc'] = np.where((bank_data['LowDoc'] == np.nan) & (bank_data['DisbursementGross'] >= 150000),0,bank_data.LowDoc)

bank_data = bank_data[(bank_data['LowDoc'] == 0) | (bank_data['LowDoc'] == 1)]
bank_data.LowDoc.value_counts()

bank_data['LowDoc'].unique()


# In[ ]:


bank_data['MIS_Status'] = np.where((bank_data['MIS_Status'] == 0.0) & (bank_data['ChgOffDate'] == np.nan),0,bank_data.MIS_Status)
bank_data['MIS_Status'] = np.where((bank_data['MIS_Status'] == 1.0) & (bank_data['ChgOffDate'] != np.nan),1,bank_data.MIS_Status)

bank_data = bank_data[(bank_data['MIS_Status'] == 0) | (bank_data['MIS_Status'] == 1)]


# In[ ]:


bank_data['MIS_Status'].unique()


# In[ ]:


print(bank_data[['MIS_Status', 'ChgOffDate']].head(10))


# In[ ]:


bank_data.MIS_Status.value_counts()


# In[ ]:


bank_data.isnull().sum()


# In[ ]:


bank_data = bank_data.drop(axis=1, columns=['Name', 'City', 'Bank', 'Zip', 'DisbursementDate','FranchiseCode', 'CCSC', 'ChgOffDate','ChgOffPrinGr'])


# In[ ]:


bank_data.isnull().sum()


# In[ ]:


bank_data.dropna(subset=['DisbursementFY'], inplace=True)


# In[ ]:


bank_data.isnull().sum()


# In[ ]:


bank_data.dtypes


# In[ ]:


bank_data = bank_data.astype({'UrbanRural': 'int64', 
                    'RevLineCr': 'int64', 
                    'LowDoc':'int64', 
                    'MIS_Status':'int64'})


# In[ ]:


# Create StateSame flag field which identifies where the business State is the same as the BankState
bank_data['IsSameState'] = np.where(bank_data['State'] == bank_data['BankState'], 1, 0)


# In[ ]:


bank_data = bank_data.drop(axis=1, columns=['State','BankState', 'ApprovalDate'])


# In[ ]:


bank_data.shape


# In[ ]:


## Handling Outliers
def limit(i):
    Q1 = bank_data[i].quantile(0.25)
    Q3 = bank_data[i].quantile(0.75)
    IQR = Q3 - Q1
    
    #determine the usual upper and extreme upper limits
    lower_limit = bank_data[i].quantile(0.25) - (IQR * 1.5)
    lower_limit_extreme = bank_data[i].quantile(0.25) - (IQR * 3)
    upper_limit = bank_data[i].quantile(0.75) + (IQR * 1.5)
    upper_limit_extreme = bank_data[i].quantile(0.75) + (IQR * 3)
    print('Lower Limit:', lower_limit)
    print('Lower Limit Extreme:', lower_limit_extreme)
    print('Upper Limit:', upper_limit)
    print('Upper Limit Extreme:', upper_limit_extreme)


# In[ ]:


#Calculates percent outliers from data   
def percent_outliers(i):
    Q1 = bank_data[i].quantile(0.25)
    Q3 = bank_data[i].quantile(0.75)
    IQR = Q3 - Q1
    
    #determine the usual upper and extreme upper limits
    lower_limit = bank_data[i].quantile(0.25) - (IQR * 1.5)
    lower_limit_extreme = bank_data[i].quantile(0.25) - (IQR * 3)
    upper_limit = bank_data[i].quantile(0.75) + (IQR * 1.5)
    upper_limit_extreme = bank_data[i].quantile(0.75) + (IQR * 3)
    #see the percentage of outliers to the total data
    print('Lower Limit: {} %'.format(bank_data[(bank_data[i] >= lower_limit)].shape[0]/ bank_data.shape[0]*100))
    print('Lower Limit Extereme: {} %'.format(bank_data[(bank_data[i] >= lower_limit_extreme)].shape[0]/bank_data.shape[0]*100))
    print('Upper Limit: {} %'.format(bank_data[(bank_data[i] >= upper_limit)].shape[0]/ bank_data.shape[0]*100))
    print('Upper Limit Extereme: {} %'.format(bank_data[(bank_data[i] >= upper_limit_extreme)].shape[0]/bank_data.shape[0]*100))


# In[ ]:


#We check the BalanceGross column outliers
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['DisbursementGross'])
plt.title('DisbursementGross Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


#we will check the limit of outliers and what percentage of our data exceeds the limit
#from sympy import limit
print(limit('DisbursementGross'))
print('-'*50)
print(percent_outliers('DisbursementGross'))


# In[ ]:


#because there are 10% of the amount of data that we have, so I try to change the data using
#log transformation, because if outliers are removed very much data is lost (10%)
bank_data['DisbursementGross'] = np.log(bank_data['DisbursementGross'])
bank_data['DisbursementGross'].skew()


# In[ ]:


#we will check the limit outliers and what percentage of our data exceeds the limit
print(limit('DisbursementGross'))
print('-'*50)
print(percent_outliers('DisbursementGross'))


# In[ ]:


#it turns out there are still about .23% outliers, because the amount is relatively small, then we just drop it
outliers1_drop = bank_data[(bank_data['DisbursementGross'] > 14.3)].index
bank_data.drop(outliers1_drop, inplace=True)


# In[ ]:


#we check again whether there are still outliers
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['DisbursementGross'])
plt.title('DisbursementGross Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


#we check the GrAppv column for outliers
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['GrAppv'])
plt.title('GrAppv Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


#we will check the limit of outliers and what percentage of our data exceeds the limit
print(limit('GrAppv'))
print('-'*50)
print(percent_outliers('GrAppv'))


# In[ ]:


bank_data['GrAppv'] = np.log(bank_data['GrAppv'])
bank_data['GrAppv'].skew()


# In[ ]:


#we will check the limit of outliers and what percentage of our data exceeds the limit
print(limit('GrAppv'))
print('-'*50)
print(percent_outliers('GrAppv'))


# In[ ]:


outliers2_drop = bank_data[(bank_data['GrAppv'] >14.15)].index
bank_data.drop(outliers2_drop, inplace=True)


# In[ ]:


#we check again in the GrAppv column whether there are still outliers
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['GrAppv'])
plt.title('GrAppv Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


#we check ouliers in the NoEmp column
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['NoEmp'])
plt.title('NoEmp Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


#we will check the limit of outliers and what percentage of our data exceeds the limit
print(limit('NoEmp'))
print('-'*50)
print(percent_outliers('NoEmp'))


# In[ ]:


#in the NoEmp column, there is input 0, aka I consider this an error, input, because it might not be a company
#do not have employees
wrong_input = bank_data[(bank_data['NoEmp'] == 0)].index
bank_data.drop(wrong_input, inplace=True)


# In[ ]:


bank_data.NoEmp.unique()


# In[ ]:


f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['NoEmp'])
plt.title('NoEmp Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


## transform then check for NoEmp
bank_data['NoEmp'] = np.log(bank_data['NoEmp'])
bank_data['NoEmp'].skew()


# In[ ]:


f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['NoEmp'])
plt.title('NoEmp Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


print(limit('NoEmp'))
print('-'*50)
print(percent_outliers('NoEmp'))


# In[ ]:


outliers3_drop = bank_data[(bank_data['NoEmp'] >4.15)].index
bank_data.drop(outliers3_drop, inplace=True)


# In[ ]:


f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['NoEmp'])
plt.title('NoEmp Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


#we check outliers in the Term column
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=bank_data['Term'])
plt.title('Term Outliers', fontsize=20)
plt.xlabel('Total', fontsize=15)


# In[ ]:


bank_data.dtypes


# In[ ]:


## Feature engneering
# choosing the best possible input combnation for a better model
y = bank_data['MIS_Status']
X = bank_data.drop(columns=['MIS_Status'], axis=1)


# In[ ]:


# choosing the best possible input combnation for a better model
y = bank_data['MIS_Status']
X = bank_data.drop(columns=['MIS_Status'], axis=1)


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


#we try to use fetaure importance on the XGboost model
model1 = XGBClassifier()
model1.fit(X,y)


# In[ ]:





# In[ ]:


#We visualize important features
feat_importances = pd.Series(model1.feature_importances_, index=X.columns)
f, ax = plt.subplots(figsize=(16,9))
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance', fontsize=20)
plt.ylabel('Features', fontsize=15)
plt.xlabel('Score', fontsize=15)
plt.show()


# In[ ]:


#Based on the feature selection above, we will select these features and discard them
# features that are not relevant to the target
bank_data = bank_data[['NewExist','ApprovalFY','UrbanRural', 'GreatRecession', 'DisbursementFY','Term','IsSameState','SBA_Appv','RevLineCr','RetainedJob', 'MIS_Status']]
bank_data.shape


# In[ ]:


print(bank_data.MIS_Status.value_counts())
print('-'*50)
print('MIS_Status (0): {} %'.format(bank_data[(bank_data['MIS_Status'] == 0)].shape[0]/bank_data.shape[0]*100))
print('MIS_Status (1): {} %'.format(bank_data[(bank_data['MIS_Status'] == 1)].shape[0]/bank_data.shape[0]*100))


# In[ ]:


sns.countplot("MIS_Status",data=bank_data)


# In[ ]:


y = bank_data['MIS_Status']
X = bank_data.drop(columns=['MIS_Status'], axis=1)
scale = StandardScaler()  ## scaling the data using z values. Standardization scales each input variable separately by subtracting the mean (called centering) and dividing by the standard deviation to shift the distribution to have a mean of zero and a standard deviation of one.
X_scaled = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=27, stratify=y) #


# In[ ]:


over = SMOTE(sampling_strategy='minority')
under = RandomUnderSampler()
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

#now we fit into our data training
X_train, y_train = pipeline.fit_resample(X_train, y_train)


# In[ ]:


## MODELLING
def model_eval(algo,X_train,y_train,X_test,y_test):
    algo.fit(X_train,y_train)
    y_train_ypred = algo.predict(X_train)
    y_train_prob = algo.predict_proba(X_train)[:,-1]  ## proba usedto get the percentile

    #TEST

    y_test_ypred = algo.predict(X_test)
    y_test_prob = algo.predict_proba(X_test)[:,-1]  ## proba usedto get the percentile
    y_probas = algo.predict_proba(X_test)
    
    #Confussion Matrix
    plot_confusion_matrix(algo, X_test, y_test, cmap=plt.cm.Blues)
    plt.show() 
    print('='*100)
    print('Classification Report: \n', classification_report(y_test, y_test_ypred, digits=3))
    print('='*100)
    
    #ROC Curve
    #fpr,tpr,thresholds = roc_curve(y_test,y_test_prob)
    skplt.metrics.plot_roc(y_test, y_probas,figsize=(16,9) )
    
    #PR Curve
    skplt.metrics.plot_precision_recall(y_test, y_probas, figsize=(16,9))
    plt.show()


# In[ ]:


#Using Logistic Regression
##lr = LogisticRegression()
#model_eval(lr,X_train,y_train,X_test,y_test)


# In[ ]:


#nb = GaussianNB()
#model_eval(nb,X_train,y_train,X_test,y_test)


# In[ ]:


#knn = KNeighborsClassifier()
#model_eval(knn,X_train,y_train,X_test,y_test)


# In[ ]:


#rf = RandomForestClassifier()
#model_eval(rf,X_train,y_train,X_test,y_test)


# In[ ]:


xgb = XGBClassifier()
model_eval(xgb,X_train,y_train,X_test,y_test)


# In[ ]:


#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators=100, random_state=0)
#model_eval(clf,X_train,y_train,X_test,y_test)


# In[ ]:


###********************DEPLOYMENT******************###

#Fitting training data
xgb.fit(X_train,y_train)


# In[ ]:


#Saving model to disk
pickle.dump(xgb, open('MODEL.pkl', 'wb'))

#loading model
model=pickle.load(open('MODEL.pkl', 'rb'))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 1. features engeneering : Missing Value Imputation¶
# 2. fill 90% NA to "Nona"
# 3. lable encoding 
# 4. 

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew 
#read train data
train=pd.read_csv('/Users/mirabooboo/Desktop/Kaggle/train.csv')
test=pd.read_csv('/Users/mirabooboo/Desktop/Kaggle/test.csv')
#show data
print(train.head(3))
#check data
print (train.info)
print (train.describe())

#Visualize correlation : all variables 
corr_plt = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr_plt , vmax=0.9, cmap="viridis", square=True)
plt.title('Correlation between all features');

#Visualize correlation : important feature with salesprice r>0.5
corrMatrix=train[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))
sns.heatmap(corrMatrix, vmax=0.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between important features');


# use scatter to see we can catch up some outliner 
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=12)
plt.xlabel('GrLivArea', fontsize=12)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=12)
plt.xlabel('TotalBsmtSF', fontsize=12)
plt.show()

# check target variable 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# sales price is right skew so here take log to normallize
train["SalePrice_log"] = np.log1p(train["SalePrice"])
train["SalePrice_log"].head(3)

#combine train and test for feature engineering 
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# total data is 2919 observations
all_data.head(5)

# check missing value 
result=all_data.isnull().sum()/len(all_data)*100
result=result.drop(result[result == 0].index).sort_values(ascending=False)[:30]
missing= pd.DataFrame({'Missing_Ratio' :result})
m=missing.head(20)
print(m)

#Visualize missing value percentage 
fig,ax = plt.subplots()
plt.xticks(rotation='90')
sns.barplot(x=result.index, y=result, color='turquoise')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=12)
plt.show()

# replace the 8 variables missing values with "None" 
# 1. PoolQC: Pool quality 'NA' means no pool
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# 2. MiscFeature :Miscellaneous feature not covered in other categories 
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
# 3. Alley : 'NA' means No alley access
all_data["Alley"] = all_data["Alley"].fillna("None")
# 4. FireplaceQu: 'NA' means No fireplace   
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# 5. Fence : 'NA' means No fence
all_data["Fence"] = all_data["Fence"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
    
all_data["LotFrontage"].describe()
all_data["LotFrontage"].median()
#median of LotFrontage is 68 and mean of LotFrontage is 69.3
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.mean()))

result1=all_data.isnull().sum()/len(all_data)*100
result1=result1.drop(result1[result1 == 0].index).sort_values(ascending=False)[:30]
missing1= pd.DataFrame({'Missing_Ratio' :result1})
m1=missing1.head(20)
print(m1)

# logic of fillmissing value with 0 is no graage=no car
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


#basement releated variables, NA means there is no basement 
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)    

result1=all_data.isnull().sum()/len(all_data)*100
result1=result1.drop(result1[result1 == 0].index).sort_values(ascending=False)[:30]
missing1= pd.DataFrame({'Missing_Ratio' :result1})
m1=missing1.head(20)
print(m1) 
#### only 10 variables with missing value <1%

# Transforming some numerical variables that are categorical variables   
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'Functional', 'Fence', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for i in cols:
    new_label= LabelEncoder() 
    new_label.fit(list(all_data[i].values)) 
    all_data[i] = new_label.transform(list(all_data[i].values))
       
print('all_data: {}'.format(all_data.shape))


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice_log


#modeling part 
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


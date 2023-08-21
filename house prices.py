#!/usr/bin/env python
# coding: utf-8

# # House Prices Prediction

# ## Import the library

# In[291]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import os
for dirname, _, filenames in os.walk('C:/Users/qqcom/Downloads/house-prices-advanced-regression-techniques'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Load the dataset

# In[234]:


train = pd.read_csv('C:/Users/qqcom/Downloads/house-prices-advanced-regression-techniques/train.csv')
train


# In[235]:


train.info()


# In[236]:


null_num = train.isnull().sum()
null_num


# ## SalePrice Distribution

# In[237]:


train['SalePrice'].describe()


# In[238]:


plt.figure(dpi=200, figsize=(20, 8))
sns.set()
SalePrice_hist = sns.histplot(train['SalePrice'], kde=True, bins=120)
SalePrice_hist.set_title('SalePrice_hist')
plt.savefig('C:/Users/qqcom/Downloads/house-prices-advanced-regression-techniques/SalePrice_histfig2')


# In[244]:


train = train.drop('Id',axis=1)


# ## Numerial data distribution

# In[279]:


#筛选出数字类型的列
train_num = train.select_dtypes(exclude='object')
train_num


# In[280]:


train_num.isnull().sum()


# In[281]:


train_num = train_num.drop(columns=['LotFrontage','MasVnrArea' ,'GarageYrBlt'],axis=1)
train_num


# In[282]:


train_num.hist(figsize=(25,25))
plt.savefig('C:/Users/qqcom/Downloads/house-prices-advanced-regression-techniques/Train_num_hist')


# ## Encode columns of non-numeric types

# In[241]:


train_obj = train.select_dtypes(include='object')
train_obj


# In[171]:


from sklearn.preprocessing import LabelEncoder


# In[172]:


#label_encoder = LabelEncoder()
#train_obj_encoder = train_obj.apply(label_encoder.fit_transform)
#train_obj_encoder


# ### View the mapping relationship

# In[296]:


train_obj = train.select_dtypes(include='object')
label_encoder_dict = {}
for col in train_obj.columns:
    label_encoder_temp = LabelEncoder()
    train_obj[col] = label_encoder_temp.fit_transform(train_obj[col])
    unique_list = train_obj[col].unique()
    
    unique_dict = {}
    for v in unique_list:
        unique_dict[v] = label_encoder_temp.inverse_transform(np.array([v]))[0]
    
    label_encoder_dict[col] = unique_dict


# In[229]:


for key, value in label_encoder_dict.items():
    print(key, value)


# ## Prepare the data

# In[297]:


train_total = pd.concat([train_num, train_obj], axis=1)
train_total


# In[298]:


from sklearn.model_selection import train_test_split

X = train_total.drop('SalePrice', axis=1)
y = train_total['SalePrice']

X_train, X_test,y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.2, shuffle=True)


# In[299]:


print("The size of X_train is ", X_train.shape)
print("The size of X_test is ", X_test.shape)
print("The size of y_train is ", y_train.shape)
print("The size of y_test is ", y_test.shape)


# ## Select a Model and Train the model

# In[300]:


from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
print(train_score)


# In[301]:


from sklearn.metrics import mean_squared_error , r2_score

mse = mean_squared_error(y_test, model.predict(X_test ))
print(mse)


# In[302]:


r2 = r2_score(y_test, model.predict(X_test ))
print(r2)


# In[303]:


from math import sqrt
rmse = sqrt(mse)
print(rmse)


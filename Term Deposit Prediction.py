#!/usr/bin/env python
# coding: utf-8

# In[349]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[350]:


train = pd.read_csv('C:/Users/qqcom/Downloads/term_deposit_prediction/train.csv')
train


# In[351]:


train.info()


# In[194]:


train.shape


# In[195]:


train.columns


# In[196]:


train.isnull().sum()


# ## Categorical Features

# In[197]:


train_obj = train.select_dtypes(include='object')


# In[8]:


for col in train_obj.columns:
    print(col, end=': ')
    print(train_obj[col].unique())


# In[20]:


cg_features = [col for col in train_obj.columns if col not in ['subscribed']]
cg_features


# In[87]:


plt.figure(figsize=(40,50))
plotnum = 1
for cgf in cg_features:
    ax = plt.subplot(4,3,plotnum)
    sns.countplot(x=train_obj[cgf])
    sns.set(context='notebook', style='ticks', font_scale=3)
    plt.xticks(rotation=45)
    plt.xlabel(cgf)
    plt.title(cgf)
    sns.set_palette("Set3")
    plotnum += 1
plt.tight_layout()
plt.show()


# ### Relation with target feature

# In[96]:


train_sub_yes = train_obj[train_obj['subscribed'] == 'yes']
train_sub_yes


# In[162]:


plt.figure(figsize=(12,14))
plotnum = 1
for feature in cg_features:
    ax = plt.subplot(4,3,plotnum)
    plt.pie(train_sub_yes[feature].value_counts(), labels =train_sub_yes[feature].unique(),
            startangle = 90,counterclock = False )
    sns.set_palette("Blues")
    plt.xlabel(feature)
    plotnum += 1
plt.tight_layout()


# In[163]:


plt.figure(figsize=(20,20))
for feature in cg_features:
    sns.set()
    sns.catplot( x='subscribed', col=feature,kind='count', data=train_obj)
    
plt.show()


# ## Numerial features

# In[10]:


train_num = train.select_dtypes(exclude='object')
train_num


# In[11]:


num_features = [col for col in train_num.columns if (col not in ['ID'])]
print(num_features)


# In[65]:


plt.figure(figsize=(20,20))
plotnum = 1
for feature in num_features:
    ax = plt.subplot(4,3,plotnum)
    sns.distplot(train[feature])
    sns.set(context='notebook', style='ticks', font_scale=1.5)
    plt.xlabel(feature)
    plotnum += 1
plt.tight_layout()


# #### Visualizing the relationship between all numerical features using Heatmap

# In[150]:


matrix = train.corr()
sns.heatmap(matrix, square=True, cmap="Reds",center=0.3)


# ## Cleaning data

# In[283]:


train['subscribed'].value_counts(normalize=True)


# In[284]:


train['default'].value_counts(normalize=True) #remove 'default'


# In[352]:


train.drop(['ID', 'default'], axis=1,inplace=True)


# In[286]:


train['pdays'].value_counts(normalize=True) #remove 'pdays'


# In[353]:


train.drop(['pdays'],axis=1,inplace=True)
train


# In[288]:


sns.catplot(x='subscribed',y='balance',data=train,kind='violin')


# In[210]:


sns.catplot(x='subscribed',y='duration',data=train,kind='bar')


# In[289]:


train.groupby(['subscribed','campaign'],sort=True).size()


# In[354]:


cate_columns = ['job','marital','education','contact','month','poutcome']
bool_columns=['housing','loan','subscribed']


# In[355]:


from sklearn.preprocessing import LabelEncoder

label_encoder_dict = {}
for col in cate_columns:
    label_encoder_temp = LabelEncoder()
    train[col] = label_encoder_temp.fit_transform(train[col])
    unique_list = train[col].unique()
    unique_list.sort()
    
    unique_dict = {}
    for v in unique_list:
        unique_dict[v] = label_encoder_temp.inverse_transform(np.array([v]))[0]
    
    label_encoder_dict[col] = unique_dict


# In[356]:


for key, value in label_encoder_dict.items():
    print(key, value)


# In[347]:


print(bool_columns)


# In[357]:


for col in bool_columns:
    train[col+'_new'] = train[col].apply(lambda x : 1 if x=='yes' else 0)
    train.drop(col,axis=1,inplace=True)


# In[300]:


train


# ## Splitting the dataset

# In[336]:


X = train.drop(['subscribed_new'],axis=1)
y = train['subscribed_new']


# In[337]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=True)


# In[338]:


print("The size of X_train is ", X_train.shape)
print("The size of X_test is ", X_test.shape)
print("The size of y_train is ", y_train.shape)
print("The size of y_test is ", y_test.shape)


# ## Model selection

# In[316]:


from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn import metrics

lg = LogisticRegression()
lg.fit(X_train, y_train)
print(lg.coef_)


# In[310]:


y_predict = lg.predict(X_test)
print("准确率：", lg.score(X_test, y_test))


# In[405]:


RFC_score = cross_val_score(estimator=RandomForestClassifier(),
                            X=X_train,y=y_train, cv=5,scoring='accuracy' )
print(RFC_score)


# In[406]:


print(RFC_score.mean())


# ## Submission

# In[392]:


test=pd.read_csv('C:/Users/qqcom/Downloads/term_deposit_prediction/test.csv')
test_original = test.copy()


# In[384]:


test.isnull().sum()


# In[393]:


test.drop(['ID','default','pdays'],axis=1,inplace=True)
print(test)


# In[394]:


cate_columns = ['job','marital','education','contact','month','poutcome']
bool_columns=['housing','loan']

from sklearn.preprocessing import LabelEncoder

label_encoder_dict = {}
for col in cate_columns:
    label_encoder_temp = LabelEncoder()
    test[col] = label_encoder_temp.fit_transform(test[col])
    unique_list = test[col].unique()
    unique_list.sort()
    
    unique_dict = {}
    for v in unique_list:
        unique_dict[v] = label_encoder_temp.inverse_transform(np.array([v]))[0]
    
    label_encoder_dict[col] = unique_dict
    
for key, value in label_encoder_dict.items():
    print(key, value)


# In[395]:


print(test)


# In[396]:


for col in bool_columns:
    test[col+'_new'] = test[col].apply(lambda x : 1 if x=='yes' else 0)
    test.drop(col,axis=1,inplace=True)


# In[397]:


print(test)


# In[398]:


pred_test = lg.predict(test)
print(pred_test)


# In[399]:


submission = pd.DataFrame(pred_test,columns = ['subscribed'] )
submission = pd.concat([test_original['ID'], submission], axis = 1)
print(submission)


# In[404]:


submission.to_csv('C:/Users/qqcom/Downloads/term_deposit_prediction/submission.csv')


# In[401]:


submission['subscribed'].value_counts()


# In[402]:


submission['subscribed'].value_counts(normalize=True)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score

plt.rcParams['font.size'] = 18

seed=1000
np.random.seed(seed)

df=pd.read_csv('Chapter5_Geologic_DataSet.csv')
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(),annot=True)
plt.title('Feature Corrletaion Matrix')
#plt.tight_layout()
plt.show()


# In[3]:


df.drop(['TOC (%)','Matrix Perm (nd)'],axis=1,inplace=True)

y=df['Aroot(K)']
x=df.drop(['Aroot(K)'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
rf=RandomForestRegressor(n_estimators=5000,criterion='mse',min_samples_split=4,min_samples_leaf=2)
rf.fit(x_train,y_train)
y_pred_train=rf.predict(x_train)
y_pred_test=rf.predict(x_test)

print('R^2 for training data is %f' %metrics.r2_score(y_train,y_pred_train))
print('R^2 for testing data is %f' %metrics.r2_score(y_test,y_pred_test))


# In[4]:


cross_val_r2=cross_val_score(rf,x,y,cv=5,scoring='r2')
print('R^2 for five-fold cross validation is %f' %cross_val_r2.mean())


# In[5]:


plt.figure(figsize=(10,8))
plt.scatter(y_pred_test,y_test)
plt.xlabel('Aroot(K) Testing Prediction')
plt.ylabel('Aroot(K) Testing Actual')
plt.title('Aroot(K) Testing Actual vs Prediction')
plt.show()


# In[ ]:





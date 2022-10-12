#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor

seed=1000
np.random.seed(seed)

df=pd.read_csv('USROP_A 4 N-SH_F-15Sd.csv')
print(df.head())

X=df.drop(['Rate of Penetration m/h'],axis=1)
Y=df['Rate of Penetration m/h']

increment=577
train_depth=increment
test_depth=train_depth+increment

y_test_pred_all=[]

while True:
    if train_depth>df.index.max()+1:
        break
    if test_depth>df.index.max()+1:
        test_depth=df.index.max()+1
    
    x_train=X[0:train_depth]
    y_train=Y[0:train_depth]
    x_test=X[train_depth:test_depth]
    y_test=Y[train_depth:test_depth]
    
    xgb=XGBRegressor()
    xgb.fit(x_train,y_train)
    y_test_pred=xgb.predict(x_test)
    
    y_test_pred_all.append(y_test_pred)
    
    train_depth=train_depth+increment
    test_depth=train_depth+increment


# In[22]:


y_test_pred_all_array=np.concatenate(y_test_pred_all,axis=0)

y_test_all=np.array(Y[577:])

print(r2_score(y_test_all,y_test_pred_all_array))
print(mean_absolute_error(y_test_all,y_test_pred_all_array))


# In[ ]:





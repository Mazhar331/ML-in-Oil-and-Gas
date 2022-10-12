#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor

seed=1000
np.random.seed(seed)

df=pd.read_csv('USROP_A 4 N-SH_F-15Sd.csv')
print(df.head())

df.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

scaler=preprocessing.MinMaxScaler(feature_range=(0,1))

increment=577
train_depth=increment
test_depth=train_depth+increment

y_test_pred_all=[]

while True:
    if train_depth>df.index.max()+1:
        break
    if test_depth>df.index.max()+1:
        test_depth=df.index.max()+1
    
    df_train=df[0:train_depth]
    scaler.fit(df_train)
    df_train_scaled=scaler.transform(df_train)
    df_train_scaled=pd.DataFrame(df_train_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])
    
    x_train_scaled=df_train_scaled.drop(['Rate of Penetration m/h'],axis=1)
    y_train_scaled=df_train_scaled['Rate of Penetration m/h']
    
    xgb=XGBRegressor()
    xgb.fit(x_train_scaled,y_train_scaled)
    
    df_train_test=df[0:test_depth]
    scaler.fit(df_train_test)
    df_train_test_scaled=scaler.transform(df_train_test)
    df_train_test_scaled=pd.DataFrame(df_train_test_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])
    
    x_train_test_scaled=df_train_test_scaled.drop(['Rate of Penetration m/h'],axis=1)
    y_train_test_scaled=df_train_test_scaled['Rate of Penetration m/h']
    
    x_test_scaled=x_train_test_scaled[train_depth:test_depth]
    y_test_scaled=y_train_test_scaled[train_depth:test_depth]
    
    y_test_pred_scaled=xgb.predict(x_test_scaled)
    
    y_test_pred=y_test_pred_scaled*(max(df_train_test['Rate of Penetration m/h'])-min(df_train_test['Rate of Penetration m/h']))+min(df_train_test['Rate of Penetration m/h'])
    
    y_test_pred_all.append(y_test_pred)
    
    train_depth=train_depth+increment
    test_depth=train_depth+increment
    
    '''
    train_data=df[0:train_depth]
    
    x_train=train_data.drop(['Rate of Penetration m/h'],axis=1)
    y_train=train_data['Rate of Penetration m/h']
    
    test_data=df[train_depth:test_depth]
    
    x_test=test_data.drop(['Rate of Penetration m/h'],axis=1)
    y_test=test_data['Rate of Penetration m/h']
    
    xgb=XGBRegressor()
    xgb.fit(x_train,y_train)
    y_test_pred=xgb.predict(x_test)
    
    y_test_pred_all.append(y_test_pred)
    
    train_depth=train_depth+increment
    test_depth=train_depth+increment
    '''


# In[35]:


#y_test_all=np.array(df['Rate of Penetration m/h'][577:])

y_test_pred_all_array=np.concatenate(y_test_pred_all,axis=0)

print(r2_score(y_test_all,y_test_pred_all_array))
print(mean_absolute_error(y_test_all,y_test_pred_all_array))


# In[33]:


import seaborn as sns

plt.hist(df['Measured Depth m'])


# In[30]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:





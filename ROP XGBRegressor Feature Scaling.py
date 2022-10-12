#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#X=df.drop(['Rate of Penetration m/h'],axis=1)
#Y=df['Rate of Penetration m/h']

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
    df_train_scaled=pd.DataFrame(df_train_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Diameter mm','Average Hookload kkgf','Hole Depth (TVD) m','USROP Gamma gAPI'])
    
    x_train_scaled=df_train_scaled.drop(['Rate of Penetration m/h'],axis=1)
    y_train_scaled=df_train_scaled['Rate of Penetration m/h']
    
    xgb=XGBRegressor()
    xgb.fit(x_train_scaled,y_train_scaled)
    
    df_train_test=df[0:test_depth]
    scaler.fit(df_train_test)
    df_train_test_scaled=scaler.transform(df_train_test)
    df_train_test_scaled=pd.DataFrame(df_train_test_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Diameter mm','Average Hookload kkgf','Hole Depth (TVD) m','USROP Gamma gAPI'])
    
    x_train_test_scaled=df_train_test_scaled.drop(['Rate of Penetration m/h'],axis=1)
    y_train_test_scaled=df_train_test_scaled['Rate of Penetration m/h']
    
    x_test_scaled=x_train_test_scaled[train_depth:test_depth]
    y_test_scaled=y_train_test_scaled[train_depth:test_depth]
    
    y_test_pred_scaled=xgb.predict(x_test_scaled)
    
    y_test_pred=y_test_pred_scaled*(max(df_train_test['Rate of Penetration m/h'])-min(df_train_test['Rate of Penetration m/h']))+min(df_train_test['Rate of Penetration m/h'])
    
    y_test_pred_all.append(y_test_pred)
    
    train_depth=train_depth+increment
    test_depth=train_depth+increment


# In[2]:


y_test_all=np.array(df['Rate of Penetration m/h'][577:])

y_test_pred_all_array=np.concatenate(y_test_pred_all,axis=0)

print(r2_score(y_test_all,y_test_pred_all_array))
print(mean_absolute_error(y_test_all,y_test_pred_all_array))


# In[3]:


plt.scatter(y_test_all,y_test_pred_all_array)
plt.show()


# In[4]:


x=df['Measured Depth m'][577:]

plt.plot(x,y_test_pred_all_array,label='Prediction')
plt.plot(x,y_test_all,label='Actual')
plt.legend()
plt.show()


# In[5]:


df.describe()


# In[6]:


y_test_dict={'True':y_test_all,'Prediction':y_test_pred_all_array}
y_test_df=pd.DataFrame(y_test_dict)
y_test_df=y_test_df[(y_test_df['True']>df['Rate of Penetration m/h'].mean()-3*df['Rate of Penetration m/h'].std())&(y_test_df['True']<df['Rate of Penetration m/h'].mean()+3*df['Rate of Penetration m/h'].std())]

print(r2_score(y_test_df['True'],y_test_df['Prediction']))
print(mean_absolute_error(y_test_df['True'],y_test_df['Prediction']))


# In[7]:


plt.scatter(y_test_df['True'],y_test_df['Prediction'])


# In[ ]:





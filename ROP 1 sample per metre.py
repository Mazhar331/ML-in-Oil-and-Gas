#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor

seed=1000
np.random.seed(seed)

scaler=preprocessing.MinMaxScaler(feature_range=(0,1))

df1=pd.read_csv('USROP_A 4 N-SH_F-15Sd modified.csv')
df1.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df1=df1[(df1['Weight on Bit kkgf']<(df1['Weight on Bit kkgf'].mean()+3*df1['Weight on Bit kkgf'].std()))&(df1['Weight on Bit kkgf']>(df1['Weight on Bit kkgf'].mean()-3*df1['Weight on Bit kkgf'].std()))&(df1['Average Standpipe Pressure kPa']<(df1['Average Standpipe Pressure kPa'].mean()+3*df1['Average Standpipe Pressure kPa'].std()))&(df1['Average Standpipe Pressure kPa']>(df1['Average Standpipe Pressure kPa'].mean()-3*df1['Average Standpipe Pressure kPa'].std()))&(df1['Average Surface Torque kN.m']<(df1['Average Surface Torque kN.m'].mean()+3*df1['Average Surface Torque kN.m'].std()))&(df1['Average Surface Torque kN.m']>(df1['Average Surface Torque kN.m'].mean()-3*df1['Average Surface Torque kN.m'].std()))&(df1['Rate of Penetration m/h']<(df1['Rate of Penetration m/h'].mean()+3*df1['Rate of Penetration m/h'].std()))&(df1['Rate of Penetration m/h']>(df1['Rate of Penetration m/h'].mean()-3*df1['Rate of Penetration m/h'].std()))&(df1['Average Rotary Speed rpm']<(df1['Average Rotary Speed rpm'].mean()+3*df1['Average Rotary Speed rpm'].std()))&(df1['Average Rotary Speed rpm']>(df1['Average Rotary Speed rpm'].mean()-3*df1['Average Rotary Speed rpm'].std()))&(df1['Mud Flow In L/min']<(df1['Mud Flow In L/min'].mean()+3*df1['Mud Flow In L/min'].std()))&(df1['Mud Flow In L/min']>(df1['Mud Flow In L/min'].mean()-3*df1['Mud Flow In L/min'].std()))&(df1['Mud Density In g/cm3']<(df1['Mud Density In g/cm3'].mean()+3*df1['Mud Density In g/cm3'].std()))&(df1['Mud Density In g/cm3']>(df1['Mud Density In g/cm3'].mean()-3*df1['Mud Density In g/cm3'].std()))&(df1['Average Hookload kkgf']<(df1['Average Hookload kkgf'].mean()+3*df1['Average Hookload kkgf'].std()))&(df1['Average Hookload kkgf']>(df1['Average Hookload kkgf'].mean()-3*df1['Average Hookload kkgf'].std()))&(df1['USROP Gamma gAPI']<(df1['USROP Gamma gAPI'].mean()+3*df1['USROP Gamma gAPI'].std()))&(df1['USROP Gamma gAPI']>(df1['USROP Gamma gAPI'].mean()-3*df1['USROP Gamma gAPI'].std()))]
scaler.fit(df1)
df_train_scaled=scaler.transform(df1)
df_train_scaled=pd.DataFrame(df_train_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])

df2=pd.read_csv('USROP_A 5 N-SH-F-5d modified.csv')
df2.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df2=df2[(df2['Weight on Bit kkgf']<(df2['Weight on Bit kkgf'].mean()+3*df2['Weight on Bit kkgf'].std()))&(df2['Weight on Bit kkgf']>(df2['Weight on Bit kkgf'].mean()-3*df2['Weight on Bit kkgf'].std()))&(df2['Average Standpipe Pressure kPa']<(df2['Average Standpipe Pressure kPa'].mean()+3*df2['Average Standpipe Pressure kPa'].std()))&(df2['Average Standpipe Pressure kPa']>(df2['Average Standpipe Pressure kPa'].mean()-3*df2['Average Standpipe Pressure kPa'].std()))&(df2['Average Surface Torque kN.m']<(df2['Average Surface Torque kN.m'].mean()+3*df2['Average Surface Torque kN.m'].std()))&(df2['Average Surface Torque kN.m']>(df2['Average Surface Torque kN.m'].mean()-3*df2['Average Surface Torque kN.m'].std()))&(df2['Rate of Penetration m/h']<(df2['Rate of Penetration m/h'].mean()+3*df2['Rate of Penetration m/h'].std()))&(df2['Rate of Penetration m/h']>(df2['Rate of Penetration m/h'].mean()-3*df2['Rate of Penetration m/h'].std()))&(df2['Average Rotary Speed rpm']<(df2['Average Rotary Speed rpm'].mean()+3*df2['Average Rotary Speed rpm'].std()))&(df2['Average Rotary Speed rpm']>(df2['Average Rotary Speed rpm'].mean()-3*df2['Average Rotary Speed rpm'].std()))&(df2['Mud Flow In L/min']<(df2['Mud Flow In L/min'].mean()+3*df2['Mud Flow In L/min'].std()))&(df2['Mud Flow In L/min']>(df2['Mud Flow In L/min'].mean()-3*df2['Mud Flow In L/min'].std()))&(df2['Mud Density In g/cm3']<(df2['Mud Density In g/cm3'].mean()+3*df2['Mud Density In g/cm3'].std()))&(df2['Mud Density In g/cm3']>(df2['Mud Density In g/cm3'].mean()-3*df2['Mud Density In g/cm3'].std()))&(df2['Average Hookload kkgf']<(df2['Average Hookload kkgf'].mean()+3*df2['Average Hookload kkgf'].std()))&(df2['Average Hookload kkgf']>(df2['Average Hookload kkgf'].mean()-3*df2['Average Hookload kkgf'].std()))&(df2['USROP Gamma gAPI']<(df2['USROP Gamma gAPI'].mean()+3*df2['USROP Gamma gAPI'].std()))&(df2['USROP Gamma gAPI']>(df2['USROP Gamma gAPI'].mean()-3*df2['USROP Gamma gAPI'].std()))]
scaler.fit(df2)
df_test_scaled=scaler.transform(df2)
df_test_scaled=pd.DataFrame(df_test_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])


y_train_scaled=df_train_scaled['Rate of Penetration m/h']
x_train_scaled=df_train_scaled.drop(['Rate of Penetration m/h'],axis=1)

y_test_scaled=df_test_scaled['Rate of Penetration m/h']
x_test_scaled=df_test_scaled.drop(['Rate of Penetration m/h'],axis=1)

rf=RandomForestRegressor(n_estimators=5000)
rf.fit(x_train_scaled,y_train_scaled)
y_pred_train_scaled=rf.predict(x_train_scaled)
y_pred_test_scaled=rf.predict(x_test_scaled)


# In[15]:


print('R^2 for training data is %f' %r2_score(y_train_scaled,y_pred_train_scaled))
print('R^2 for testing data is %f' %r2_score(y_test_scaled,y_pred_test_scaled))


# In[18]:


plt.scatter(y_test_scaled,y_pred_test_scaled)


# In[ ]:





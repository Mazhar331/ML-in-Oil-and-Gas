#!/usr/bin/env python
# coding: utf-8

# In[8]:


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

df1=pd.read_csv('USROP_A 0 N-NA_F-9_Ad.csv')
df1.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

df2=pd.read_csv('USROP_A 1 N-S_F-7d.csv')
df2.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

df3=pd.read_csv('USROP_A 2 N-SH_F-14d.csv')
df3.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

df4=pd.read_csv('USROP_A 3 N-SH-F-15d.csv')
df4.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

df5=pd.read_csv('USROP_A 4 N-SH_F-15Sd.csv')
df5.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

df6=pd.read_csv('USROP_A 5 N-SH-F-5d.csv')
df6.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

df7=pd.read_csv('USROP_A 6 N-SH_F-9d.csv')
df7.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

#df2=df2[(df2['Weight on Bit kkgf']<(df2['Weight on Bit kkgf'].mean()+3*df2['Weight on Bit kkgf'].std()))&(df2['Weight on Bit kkgf']>(df2['Weight on Bit kkgf'].mean()-3*df2['Weight on Bit kkgf'].std()))&(df2['Average Standpipe Pressure kPa']<(df2['Average Standpipe Pressure kPa'].mean()+3*df2['Average Standpipe Pressure kPa'].std()))&(df2['Average Standpipe Pressure kPa']>(df2['Average Standpipe Pressure kPa'].mean()-3*df2['Average Standpipe Pressure kPa'].std()))&(df2['Average Surface Torque kN.m']<(df2['Average Surface Torque kN.m'].mean()+3*df2['Average Surface Torque kN.m'].std()))&(df2['Average Surface Torque kN.m']>(df2['Average Surface Torque kN.m'].mean()-3*df2['Average Surface Torque kN.m'].std()))&(df2['Rate of Penetration m/h']<(df2['Rate of Penetration m/h'].mean()+3*df2['Rate of Penetration m/h'].std()))&(df2['Rate of Penetration m/h']>(df2['Rate of Penetration m/h'].mean()-3*df2['Rate of Penetration m/h'].std()))&(df2['Average Rotary Speed rpm']<(df2['Average Rotary Speed rpm'].mean()+3*df2['Average Rotary Speed rpm'].std()))&(df2['Average Rotary Speed rpm']>(df2['Average Rotary Speed rpm'].mean()-3*df2['Average Rotary Speed rpm'].std()))&(df2['Mud Flow In L/min']<(df2['Mud Flow In L/min'].mean()+3*df2['Mud Flow In L/min'].std()))&(df2['Mud Flow In L/min']>(df2['Mud Flow In L/min'].mean()-3*df2['Mud Flow In L/min'].std()))&(df2['Mud Density In g/cm3']<(df2['Mud Density In g/cm3'].mean()+3*df2['Mud Density In g/cm3'].std()))&(df2['Mud Density In g/cm3']>(df2['Mud Density In g/cm3'].mean()-3*df2['Mud Density In g/cm3'].std()))&(df2['Average Hookload kkgf']<(df2['Average Hookload kkgf'].mean()+3*df2['Average Hookload kkgf'].std()))&(df2['Average Hookload kkgf']>(df2['Average Hookload kkgf'].mean()-3*df2['Average Hookload kkgf'].std()))&(df2['USROP Gamma gAPI']<(df2['USROP Gamma gAPI'].mean()+3*df2['USROP Gamma gAPI'].std()))&(df2['USROP Gamma gAPI']>(df2['USROP Gamma gAPI'].mean()-3*df2['USROP Gamma gAPI'].std()))]
#df2=df2.reset_index(drop=True)

#print(len(df2))
#print(df2.head())

df=pd.concat([df2,df3,df4,df5,df6,df7])

df=df[(df['Weight on Bit kkgf']<(df['Weight on Bit kkgf'].mean()+3*df['Weight on Bit kkgf'].std()))&(df['Weight on Bit kkgf']>(df['Weight on Bit kkgf'].mean()-3*df['Weight on Bit kkgf'].std()))&(df['Average Standpipe Pressure kPa']<(df['Average Standpipe Pressure kPa'].mean()+3*df['Average Standpipe Pressure kPa'].std()))&(df['Average Standpipe Pressure kPa']>(df['Average Standpipe Pressure kPa'].mean()-3*df['Average Standpipe Pressure kPa'].std()))&(df['Average Surface Torque kN.m']<(df['Average Surface Torque kN.m'].mean()+3*df['Average Surface Torque kN.m'].std()))&(df['Average Surface Torque kN.m']>(df['Average Surface Torque kN.m'].mean()-3*df['Average Surface Torque kN.m'].std()))&(df['Rate of Penetration m/h']<(df['Rate of Penetration m/h'].mean()+3*df['Rate of Penetration m/h'].std()))&(df['Rate of Penetration m/h']>(df['Rate of Penetration m/h'].mean()-3*df['Rate of Penetration m/h'].std()))&(df['Average Rotary Speed rpm']<(df['Average Rotary Speed rpm'].mean()+3*df['Average Rotary Speed rpm'].std()))&(df['Average Rotary Speed rpm']>(df['Average Rotary Speed rpm'].mean()-3*df['Average Rotary Speed rpm'].std()))&(df['Mud Flow In L/min']<(df['Mud Flow In L/min'].mean()+3*df['Mud Flow In L/min'].std()))&(df['Mud Flow In L/min']>(df['Mud Flow In L/min'].mean()-3*df['Mud Flow In L/min'].std()))&(df['Mud Density In g/cm3']<(df['Mud Density In g/cm3'].mean()+3*df['Mud Density In g/cm3'].std()))&(df['Mud Density In g/cm3']>(df['Mud Density In g/cm3'].mean()-3*df['Mud Density In g/cm3'].std()))&(df['Average Hookload kkgf']<(df['Average Hookload kkgf'].mean()+3*df['Average Hookload kkgf'].std()))&(df['Average Hookload kkgf']>(df['Average Hookload kkgf'].mean()-3*df['Average Hookload kkgf'].std()))&(df['USROP Gamma gAPI']<(df['USROP Gamma gAPI'].mean()+3*df['USROP Gamma gAPI'].std()))&(df['USROP Gamma gAPI']>(df['USROP Gamma gAPI'].mean()-3*df['USROP Gamma gAPI'].std()))]
df=df.reset_index(drop=True)

Y=df['Rate of Penetration m/h']
X=df.drop(['Rate of Penetration m/h'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

rf=RandomForestRegressor(n_estimators=100,criterion='mse',min_samples_split=4,min_samples_leaf=2)
rf.fit(x_train,y_train)
y_pred_train=rf.predict(x_train)
y_pred_test=rf.predict(x_test)


# In[9]:


print(len(df))
df.describe()


# In[10]:


print('R^2 for training data is %f' %r2_score(y_train,y_pred_train))
print('R^2 for testing data is %f' %r2_score(y_test,y_pred_test))


# In[11]:


df1=df1[(df1['Weight on Bit kkgf']<(df1['Weight on Bit kkgf'].mean()+3*df1['Weight on Bit kkgf'].std()))&(df1['Weight on Bit kkgf']>(df1['Weight on Bit kkgf'].mean()-3*df1['Weight on Bit kkgf'].std()))&(df1['Average Standpipe Pressure kPa']<(df1['Average Standpipe Pressure kPa'].mean()+3*df1['Average Standpipe Pressure kPa'].std()))&(df1['Average Standpipe Pressure kPa']>(df1['Average Standpipe Pressure kPa'].mean()-3*df1['Average Standpipe Pressure kPa'].std()))&(df1['Average Surface Torque kN.m']<(df1['Average Surface Torque kN.m'].mean()+3*df1['Average Surface Torque kN.m'].std()))&(df1['Average Surface Torque kN.m']>(df1['Average Surface Torque kN.m'].mean()-3*df1['Average Surface Torque kN.m'].std()))&(df1['Rate of Penetration m/h']<(df1['Rate of Penetration m/h'].mean()+3*df1['Rate of Penetration m/h'].std()))&(df1['Rate of Penetration m/h']>(df1['Rate of Penetration m/h'].mean()-3*df1['Rate of Penetration m/h'].std()))&(df1['Average Rotary Speed rpm']<(df1['Average Rotary Speed rpm'].mean()+3*df1['Average Rotary Speed rpm'].std()))&(df1['Average Rotary Speed rpm']>(df1['Average Rotary Speed rpm'].mean()-3*df1['Average Rotary Speed rpm'].std()))&(df1['Mud Flow In L/min']<(df1['Mud Flow In L/min'].mean()+3*df1['Mud Flow In L/min'].std()))&(df1['Mud Flow In L/min']>(df1['Mud Flow In L/min'].mean()-3*df1['Mud Flow In L/min'].std()))&(df1['Mud Density In g/cm3']<(df1['Mud Density In g/cm3'].mean()+3*df1['Mud Density In g/cm3'].std()))&(df1['Mud Density In g/cm3']>(df1['Mud Density In g/cm3'].mean()-3*df1['Mud Density In g/cm3'].std()))&(df1['Average Hookload kkgf']<(df1['Average Hookload kkgf'].mean()+3*df1['Average Hookload kkgf'].std()))&(df1['Average Hookload kkgf']>(df1['Average Hookload kkgf'].mean()-3*df1['Average Hookload kkgf'].std()))&(df1['USROP Gamma gAPI']<(df1['USROP Gamma gAPI'].mean()+3*df1['USROP Gamma gAPI'].std()))&(df1['USROP Gamma gAPI']>(df1['USROP Gamma gAPI'].mean()-3*df1['USROP Gamma gAPI'].std()))]
df1=df1[(df1['Weight on Bit kkgf']>=df['Weight on Bit kkgf'].min())&(df1['Weight on Bit kkgf']<=df['Weight on Bit kkgf'].max())&(df1['Average Standpipe Pressure kPa']>=df['Average Standpipe Pressure kPa'].min())&(df1['Average Standpipe Pressure kPa']<=df['Average Standpipe Pressure kPa'].max())&(df1['Average Surface Torque kN.m']>=df['Average Surface Torque kN.m'].min())&(df1['Average Surface Torque kN.m']<=df['Average Surface Torque kN.m'].max())&(df1['Rate of Penetration m/h']>=df['Rate of Penetration m/h'].min())&(df1['Rate of Penetration m/h']<=df['Rate of Penetration m/h'].max())&(df1['Average Rotary Speed rpm']>=df['Average Rotary Speed rpm'].min())&(df1['Average Rotary Speed rpm']<=df['Average Rotary Speed rpm'].max())&(df1['Mud Flow In L/min']>=df['Mud Flow In L/min'].min())&(df1['Mud Flow In L/min']<=df['Mud Flow In L/min'].max())&(df1['Mud Density In g/cm3']>=df['Mud Density In g/cm3'].min())&(df1['Mud Density In g/cm3']<=df['Mud Density In g/cm3'].max())&(df1['Average Hookload kkgf']>=df['Average Hookload kkgf'].min())&(df1['Average Hookload kkgf']<=df['Average Hookload kkgf'].max())&(df1['USROP Gamma gAPI']>=df['USROP Gamma gAPI'].min())&(df1['USROP Gamma gAPI']<=df['USROP Gamma gAPI'].max())]
df1.describe()


# In[12]:


y_blind=df1['Rate of Penetration m/h']
x_blind=df1.drop(['Rate of Penetration m/h'],axis=1)

y_pred_blind=rf.predict(x_blind)

print('R^2 for blind data is %f' %r2_score(y_blind,y_pred_blind))
print('RMSE for blind data is %f'%mean_squared_error(y_blind,y_pred_blind,squared=False))
print('MAE for blind data is %f'%mean_absolute_error(y_blind,y_pred_blind))


# In[36]:


plt.scatter(y_blind,y_pred_blind)
plt.show()


# In[ ]:





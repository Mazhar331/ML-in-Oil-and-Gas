#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR

seed=1000
np.random.seed(seed)

df1=pd.read_csv('USROP_A 0 N-NA_F-9_Ad.csv')
df1.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df1=df1[(df1['Weight on Bit kkgf']<(df1['Weight on Bit kkgf'].mean()+3*df1['Weight on Bit kkgf'].std()))&(df1['Weight on Bit kkgf']>(df1['Weight on Bit kkgf'].mean()-3*df1['Weight on Bit kkgf'].std()))&(df1['Average Standpipe Pressure kPa']<(df1['Average Standpipe Pressure kPa'].mean()+3*df1['Average Standpipe Pressure kPa'].std()))&(df1['Average Standpipe Pressure kPa']>(df1['Average Standpipe Pressure kPa'].mean()-3*df1['Average Standpipe Pressure kPa'].std()))&(df1['Average Surface Torque kN.m']<(df1['Average Surface Torque kN.m'].mean()+3*df1['Average Surface Torque kN.m'].std()))&(df1['Average Surface Torque kN.m']>(df1['Average Surface Torque kN.m'].mean()-3*df1['Average Surface Torque kN.m'].std()))&(df1['Rate of Penetration m/h']<(df1['Rate of Penetration m/h'].mean()+3*df1['Rate of Penetration m/h'].std()))&(df1['Rate of Penetration m/h']>(df1['Rate of Penetration m/h'].mean()-3*df1['Rate of Penetration m/h'].std()))&(df1['Average Rotary Speed rpm']<(df1['Average Rotary Speed rpm'].mean()+3*df1['Average Rotary Speed rpm'].std()))&(df1['Average Rotary Speed rpm']>(df1['Average Rotary Speed rpm'].mean()-3*df1['Average Rotary Speed rpm'].std()))&(df1['Mud Flow In L/min']<(df1['Mud Flow In L/min'].mean()+3*df1['Mud Flow In L/min'].std()))&(df1['Mud Flow In L/min']>(df1['Mud Flow In L/min'].mean()-3*df1['Mud Flow In L/min'].std()))&(df1['Mud Density In g/cm3']<(df1['Mud Density In g/cm3'].mean()+3*df1['Mud Density In g/cm3'].std()))&(df1['Mud Density In g/cm3']>(df1['Mud Density In g/cm3'].mean()-3*df1['Mud Density In g/cm3'].std()))&(df1['Average Hookload kkgf']<(df1['Average Hookload kkgf'].mean()+3*df1['Average Hookload kkgf'].std()))&(df1['Average Hookload kkgf']>(df1['Average Hookload kkgf'].mean()-3*df1['Average Hookload kkgf'].std()))&(df1['USROP Gamma gAPI']<(df1['USROP Gamma gAPI'].mean()+3*df1['USROP Gamma gAPI'].std()))&(df1['USROP Gamma gAPI']>(df1['USROP Gamma gAPI'].mean()-3*df1['USROP Gamma gAPI'].std()))]

df2=pd.read_csv('USROP_A 1 N-S_F-7d.csv')
df2.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df2=df2[(df2['Weight on Bit kkgf']<(df2['Weight on Bit kkgf'].mean()+3*df2['Weight on Bit kkgf'].std()))&(df2['Weight on Bit kkgf']>(df2['Weight on Bit kkgf'].mean()-3*df2['Weight on Bit kkgf'].std()))&(df2['Average Standpipe Pressure kPa']<(df2['Average Standpipe Pressure kPa'].mean()+3*df2['Average Standpipe Pressure kPa'].std()))&(df2['Average Standpipe Pressure kPa']>(df2['Average Standpipe Pressure kPa'].mean()-3*df2['Average Standpipe Pressure kPa'].std()))&(df2['Average Surface Torque kN.m']<(df2['Average Surface Torque kN.m'].mean()+3*df2['Average Surface Torque kN.m'].std()))&(df2['Average Surface Torque kN.m']>(df2['Average Surface Torque kN.m'].mean()-3*df2['Average Surface Torque kN.m'].std()))&(df2['Rate of Penetration m/h']<(df2['Rate of Penetration m/h'].mean()+3*df2['Rate of Penetration m/h'].std()))&(df2['Rate of Penetration m/h']>(df2['Rate of Penetration m/h'].mean()-3*df2['Rate of Penetration m/h'].std()))&(df2['Average Rotary Speed rpm']<(df2['Average Rotary Speed rpm'].mean()+3*df2['Average Rotary Speed rpm'].std()))&(df2['Average Rotary Speed rpm']>(df2['Average Rotary Speed rpm'].mean()-3*df2['Average Rotary Speed rpm'].std()))&(df2['Mud Flow In L/min']<(df2['Mud Flow In L/min'].mean()+3*df2['Mud Flow In L/min'].std()))&(df2['Mud Flow In L/min']>(df2['Mud Flow In L/min'].mean()-3*df2['Mud Flow In L/min'].std()))&(df2['Mud Density In g/cm3']<(df2['Mud Density In g/cm3'].mean()+3*df2['Mud Density In g/cm3'].std()))&(df2['Mud Density In g/cm3']>(df2['Mud Density In g/cm3'].mean()-3*df2['Mud Density In g/cm3'].std()))&(df2['Average Hookload kkgf']<(df2['Average Hookload kkgf'].mean()+3*df2['Average Hookload kkgf'].std()))&(df2['Average Hookload kkgf']>(df2['Average Hookload kkgf'].mean()-3*df2['Average Hookload kkgf'].std()))&(df2['USROP Gamma gAPI']<(df2['USROP Gamma gAPI'].mean()+3*df2['USROP Gamma gAPI'].std()))&(df2['USROP Gamma gAPI']>(df2['USROP Gamma gAPI'].mean()-3*df2['USROP Gamma gAPI'].std()))]

df3=pd.read_csv('USROP_A 2 N-SH_F-14d.csv')
df3.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df3=df3[(df3['Weight on Bit kkgf']<(df3['Weight on Bit kkgf'].mean()+3*df3['Weight on Bit kkgf'].std()))&(df3['Weight on Bit kkgf']>(df3['Weight on Bit kkgf'].mean()-3*df3['Weight on Bit kkgf'].std()))&(df3['Average Standpipe Pressure kPa']<(df3['Average Standpipe Pressure kPa'].mean()+3*df3['Average Standpipe Pressure kPa'].std()))&(df3['Average Standpipe Pressure kPa']>(df3['Average Standpipe Pressure kPa'].mean()-3*df3['Average Standpipe Pressure kPa'].std()))&(df3['Average Surface Torque kN.m']<(df3['Average Surface Torque kN.m'].mean()+3*df3['Average Surface Torque kN.m'].std()))&(df3['Average Surface Torque kN.m']>(df3['Average Surface Torque kN.m'].mean()-3*df3['Average Surface Torque kN.m'].std()))&(df3['Rate of Penetration m/h']<(df3['Rate of Penetration m/h'].mean()+3*df3['Rate of Penetration m/h'].std()))&(df3['Rate of Penetration m/h']>(df3['Rate of Penetration m/h'].mean()-3*df3['Rate of Penetration m/h'].std()))&(df3['Average Rotary Speed rpm']<(df3['Average Rotary Speed rpm'].mean()+3*df3['Average Rotary Speed rpm'].std()))&(df3['Average Rotary Speed rpm']>(df3['Average Rotary Speed rpm'].mean()-3*df3['Average Rotary Speed rpm'].std()))&(df3['Mud Flow In L/min']<(df3['Mud Flow In L/min'].mean()+3*df3['Mud Flow In L/min'].std()))&(df3['Mud Flow In L/min']>(df3['Mud Flow In L/min'].mean()-3*df3['Mud Flow In L/min'].std()))&(df3['Mud Density In g/cm3']<(df3['Mud Density In g/cm3'].mean()+3*df3['Mud Density In g/cm3'].std()))&(df3['Mud Density In g/cm3']>(df3['Mud Density In g/cm3'].mean()-3*df3['Mud Density In g/cm3'].std()))&(df3['Average Hookload kkgf']<(df3['Average Hookload kkgf'].mean()+3*df3['Average Hookload kkgf'].std()))&(df3['Average Hookload kkgf']>(df3['Average Hookload kkgf'].mean()-3*df3['Average Hookload kkgf'].std()))&(df3['USROP Gamma gAPI']<(df3['USROP Gamma gAPI'].mean()+3*df3['USROP Gamma gAPI'].std()))&(df3['USROP Gamma gAPI']>(df3['USROP Gamma gAPI'].mean()-3*df3['USROP Gamma gAPI'].std()))]

df4=pd.read_csv('USROP_A 3 N-SH-F-15d.csv')
df4.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df4=df4[(df4['Weight on Bit kkgf']<(df4['Weight on Bit kkgf'].mean()+3*df4['Weight on Bit kkgf'].std()))&(df4['Weight on Bit kkgf']>(df4['Weight on Bit kkgf'].mean()-3*df4['Weight on Bit kkgf'].std()))&(df4['Average Standpipe Pressure kPa']<(df4['Average Standpipe Pressure kPa'].mean()+3*df4['Average Standpipe Pressure kPa'].std()))&(df4['Average Standpipe Pressure kPa']>(df4['Average Standpipe Pressure kPa'].mean()-3*df4['Average Standpipe Pressure kPa'].std()))&(df4['Average Surface Torque kN.m']<(df4['Average Surface Torque kN.m'].mean()+3*df4['Average Surface Torque kN.m'].std()))&(df4['Average Surface Torque kN.m']>(df4['Average Surface Torque kN.m'].mean()-3*df4['Average Surface Torque kN.m'].std()))&(df4['Rate of Penetration m/h']<(df4['Rate of Penetration m/h'].mean()+3*df4['Rate of Penetration m/h'].std()))&(df4['Rate of Penetration m/h']>(df4['Rate of Penetration m/h'].mean()-3*df4['Rate of Penetration m/h'].std()))&(df4['Average Rotary Speed rpm']<(df4['Average Rotary Speed rpm'].mean()+3*df4['Average Rotary Speed rpm'].std()))&(df4['Average Rotary Speed rpm']>(df4['Average Rotary Speed rpm'].mean()-3*df4['Average Rotary Speed rpm'].std()))&(df4['Mud Flow In L/min']<(df4['Mud Flow In L/min'].mean()+3*df4['Mud Flow In L/min'].std()))&(df4['Mud Flow In L/min']>(df4['Mud Flow In L/min'].mean()-3*df4['Mud Flow In L/min'].std()))&(df4['Mud Density In g/cm3']<(df4['Mud Density In g/cm3'].mean()+3*df4['Mud Density In g/cm3'].std()))&(df4['Mud Density In g/cm3']>(df4['Mud Density In g/cm3'].mean()-3*df4['Mud Density In g/cm3'].std()))&(df4['Average Hookload kkgf']<(df4['Average Hookload kkgf'].mean()+3*df4['Average Hookload kkgf'].std()))&(df4['Average Hookload kkgf']>(df4['Average Hookload kkgf'].mean()-3*df4['Average Hookload kkgf'].std()))&(df4['USROP Gamma gAPI']<(df4['USROP Gamma gAPI'].mean()+3*df4['USROP Gamma gAPI'].std()))&(df4['USROP Gamma gAPI']>(df4['USROP Gamma gAPI'].mean()-3*df4['USROP Gamma gAPI'].std()))]

df5=pd.read_csv('USROP_A 4 N-SH_F-15Sd.csv')
df5.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df5=df5[(df5['Weight on Bit kkgf']<(df5['Weight on Bit kkgf'].mean()+3*df5['Weight on Bit kkgf'].std()))&(df5['Weight on Bit kkgf']>(df5['Weight on Bit kkgf'].mean()-3*df5['Weight on Bit kkgf'].std()))&(df5['Average Standpipe Pressure kPa']<(df5['Average Standpipe Pressure kPa'].mean()+3*df5['Average Standpipe Pressure kPa'].std()))&(df5['Average Standpipe Pressure kPa']>(df5['Average Standpipe Pressure kPa'].mean()-3*df5['Average Standpipe Pressure kPa'].std()))&(df5['Average Surface Torque kN.m']<(df5['Average Surface Torque kN.m'].mean()+3*df5['Average Surface Torque kN.m'].std()))&(df5['Average Surface Torque kN.m']>(df5['Average Surface Torque kN.m'].mean()-3*df5['Average Surface Torque kN.m'].std()))&(df5['Rate of Penetration m/h']<(df5['Rate of Penetration m/h'].mean()+3*df5['Rate of Penetration m/h'].std()))&(df5['Rate of Penetration m/h']>(df5['Rate of Penetration m/h'].mean()-3*df5['Rate of Penetration m/h'].std()))&(df5['Average Rotary Speed rpm']<(df5['Average Rotary Speed rpm'].mean()+3*df5['Average Rotary Speed rpm'].std()))&(df5['Average Rotary Speed rpm']>(df5['Average Rotary Speed rpm'].mean()-3*df5['Average Rotary Speed rpm'].std()))&(df5['Mud Flow In L/min']<(df5['Mud Flow In L/min'].mean()+3*df5['Mud Flow In L/min'].std()))&(df5['Mud Flow In L/min']>(df5['Mud Flow In L/min'].mean()-3*df5['Mud Flow In L/min'].std()))&(df5['Mud Density In g/cm3']<(df5['Mud Density In g/cm3'].mean()+3*df5['Mud Density In g/cm3'].std()))&(df5['Mud Density In g/cm3']>(df5['Mud Density In g/cm3'].mean()-3*df5['Mud Density In g/cm3'].std()))&(df5['Average Hookload kkgf']<(df5['Average Hookload kkgf'].mean()+3*df5['Average Hookload kkgf'].std()))&(df5['Average Hookload kkgf']>(df5['Average Hookload kkgf'].mean()-3*df5['Average Hookload kkgf'].std()))&(df5['USROP Gamma gAPI']<(df5['USROP Gamma gAPI'].mean()+3*df5['USROP Gamma gAPI'].std()))&(df5['USROP Gamma gAPI']>(df5['USROP Gamma gAPI'].mean()-3*df5['USROP Gamma gAPI'].std()))]

df6=pd.read_csv('USROP_A 5 N-SH-F-5d.csv')
df6.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df6=df6[(df6['Weight on Bit kkgf']<(df6['Weight on Bit kkgf'].mean()+3*df6['Weight on Bit kkgf'].std()))&(df6['Weight on Bit kkgf']>(df6['Weight on Bit kkgf'].mean()-3*df6['Weight on Bit kkgf'].std()))&(df6['Average Standpipe Pressure kPa']<(df6['Average Standpipe Pressure kPa'].mean()+3*df6['Average Standpipe Pressure kPa'].std()))&(df6['Average Standpipe Pressure kPa']>(df6['Average Standpipe Pressure kPa'].mean()-3*df6['Average Standpipe Pressure kPa'].std()))&(df6['Average Surface Torque kN.m']<(df6['Average Surface Torque kN.m'].mean()+3*df6['Average Surface Torque kN.m'].std()))&(df6['Average Surface Torque kN.m']>(df6['Average Surface Torque kN.m'].mean()-3*df6['Average Surface Torque kN.m'].std()))&(df6['Rate of Penetration m/h']<(df6['Rate of Penetration m/h'].mean()+3*df6['Rate of Penetration m/h'].std()))&(df6['Rate of Penetration m/h']>(df6['Rate of Penetration m/h'].mean()-3*df6['Rate of Penetration m/h'].std()))&(df6['Average Rotary Speed rpm']<(df6['Average Rotary Speed rpm'].mean()+3*df6['Average Rotary Speed rpm'].std()))&(df6['Average Rotary Speed rpm']>(df6['Average Rotary Speed rpm'].mean()-3*df6['Average Rotary Speed rpm'].std()))&(df6['Mud Flow In L/min']<(df6['Mud Flow In L/min'].mean()+3*df6['Mud Flow In L/min'].std()))&(df6['Mud Flow In L/min']>(df6['Mud Flow In L/min'].mean()-3*df6['Mud Flow In L/min'].std()))&(df6['Mud Density In g/cm3']<(df6['Mud Density In g/cm3'].mean()+3*df6['Mud Density In g/cm3'].std()))&(df6['Mud Density In g/cm3']>(df6['Mud Density In g/cm3'].mean()-3*df6['Mud Density In g/cm3'].std()))&(df6['Average Hookload kkgf']<(df6['Average Hookload kkgf'].mean()+3*df6['Average Hookload kkgf'].std()))&(df6['Average Hookload kkgf']>(df6['Average Hookload kkgf'].mean()-3*df6['Average Hookload kkgf'].std()))&(df6['USROP Gamma gAPI']<(df6['USROP Gamma gAPI'].mean()+3*df6['USROP Gamma gAPI'].std()))&(df6['USROP Gamma gAPI']>(df6['USROP Gamma gAPI'].mean()-3*df6['USROP Gamma gAPI'].std()))]

df7=pd.read_csv('USROP_A 6 N-SH_F-9d.csv')
df7.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df7=df7[(df7['Weight on Bit kkgf']<(df7['Weight on Bit kkgf'].mean()+3*df7['Weight on Bit kkgf'].std()))&(df7['Weight on Bit kkgf']>(df7['Weight on Bit kkgf'].mean()-3*df7['Weight on Bit kkgf'].std()))&(df7['Average Standpipe Pressure kPa']<(df7['Average Standpipe Pressure kPa'].mean()+3*df7['Average Standpipe Pressure kPa'].std()))&(df7['Average Standpipe Pressure kPa']>(df7['Average Standpipe Pressure kPa'].mean()-3*df7['Average Standpipe Pressure kPa'].std()))&(df7['Average Surface Torque kN.m']<(df7['Average Surface Torque kN.m'].mean()+3*df7['Average Surface Torque kN.m'].std()))&(df7['Average Surface Torque kN.m']>(df7['Average Surface Torque kN.m'].mean()-3*df7['Average Surface Torque kN.m'].std()))&(df7['Rate of Penetration m/h']<(df7['Rate of Penetration m/h'].mean()+3*df7['Rate of Penetration m/h'].std()))&(df7['Rate of Penetration m/h']>(df7['Rate of Penetration m/h'].mean()-3*df7['Rate of Penetration m/h'].std()))&(df7['Average Rotary Speed rpm']<(df7['Average Rotary Speed rpm'].mean()+3*df7['Average Rotary Speed rpm'].std()))&(df7['Average Rotary Speed rpm']>(df7['Average Rotary Speed rpm'].mean()-3*df7['Average Rotary Speed rpm'].std()))&(df7['Mud Flow In L/min']<(df7['Mud Flow In L/min'].mean()+3*df7['Mud Flow In L/min'].std()))&(df7['Mud Flow In L/min']>(df7['Mud Flow In L/min'].mean()-3*df7['Mud Flow In L/min'].std()))&(df7['Mud Density In g/cm3']<(df7['Mud Density In g/cm3'].mean()+3*df7['Mud Density In g/cm3'].std()))&(df7['Mud Density In g/cm3']>(df7['Mud Density In g/cm3'].mean()-3*df7['Mud Density In g/cm3'].std()))&(df7['Average Hookload kkgf']<(df7['Average Hookload kkgf'].mean()+3*df7['Average Hookload kkgf'].std()))&(df7['Average Hookload kkgf']>(df7['Average Hookload kkgf'].mean()-3*df7['Average Hookload kkgf'].std()))&(df7['USROP Gamma gAPI']<(df7['USROP Gamma gAPI'].mean()+3*df7['USROP Gamma gAPI'].std()))&(df7['USROP Gamma gAPI']>(df7['USROP Gamma gAPI'].mean()-3*df7['USROP Gamma gAPI'].std()))]

df=pd.concat([df2,df3,df4,df5,df6,df7])
print(len(df))

Y=df['Rate of Penetration m/h']
X=df.drop(['Rate of Penetration m/h'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

'''
rf=RandomForestRegressor(n_estimators=100,criterion='mse',min_samples_split=4,min_samples_leaf=2)
rf.fit(x_train,y_train)
y_pred_train=rf.predict(x_train)
y_pred_test=rf.predict(x_test)
'''

SVM=SVR(kernel='rbf',gamma=1.5,C=5)
SVM.fit(x_train,y_train)
y_pred_train=SVM.predict(x_train)
y_pred_test=SVM.predict(x_test)


# In[4]:


print('R^2 for training data is %f' %r2_score(y_train,y_pred_train))
print('R^2 for testing data is %f' %r2_score(y_test,y_pred_test))


# In[5]:


y_blind=df1['Rate of Penetration m/h']
x_blind=df1.drop(['Rate of Penetration m/h'],axis=1)

#y_pred_blind=rf.predict(x_blind)
y_pred_blind=SVM.predict(x_blind)

print('R^2 for blind data is %f' %r2_score(y_blind,y_pred_blind))


# In[ ]:





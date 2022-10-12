#!/usr/bin/env python
# coding: utf-8

# In[101]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor

seed=1000
np.random.seed(seed)

df=pd.read_csv('USROP_A 1 N-S_F-7d.csv')

df.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)
df=df[(df['Weight on Bit kkgf']<(df['Weight on Bit kkgf'].mean()+3*df['Weight on Bit kkgf'].std()))&(df['Weight on Bit kkgf']>(df['Weight on Bit kkgf'].mean()-3*df['Weight on Bit kkgf'].std()))&(df['Average Standpipe Pressure kPa']<(df['Average Standpipe Pressure kPa'].mean()+3*df['Average Standpipe Pressure kPa'].std()))&(df['Average Standpipe Pressure kPa']>(df['Average Standpipe Pressure kPa'].mean()-3*df['Average Standpipe Pressure kPa'].std()))&(df['Average Surface Torque kN.m']<(df['Average Surface Torque kN.m'].mean()+3*df['Average Surface Torque kN.m'].std()))&(df['Average Surface Torque kN.m']>(df['Average Surface Torque kN.m'].mean()-3*df['Average Surface Torque kN.m'].std()))&(df['Rate of Penetration m/h']<(df['Rate of Penetration m/h'].mean()+3*df['Rate of Penetration m/h'].std()))&(df['Rate of Penetration m/h']>(df['Rate of Penetration m/h'].mean()-3*df['Rate of Penetration m/h'].std()))&(df['Average Rotary Speed rpm']<(df['Average Rotary Speed rpm'].mean()+3*df['Average Rotary Speed rpm'].std()))&(df['Average Rotary Speed rpm']>(df['Average Rotary Speed rpm'].mean()-3*df['Average Rotary Speed rpm'].std()))&(df['Mud Flow In L/min']<(df['Mud Flow In L/min'].mean()+3*df['Mud Flow In L/min'].std()))&(df['Mud Flow In L/min']>(df['Mud Flow In L/min'].mean()-3*df['Mud Flow In L/min'].std()))&(df['Mud Density In g/cm3']<(df['Mud Density In g/cm3'].mean()+3*df['Mud Density In g/cm3'].std()))&(df['Mud Density In g/cm3']>(df['Mud Density In g/cm3'].mean()-3*df['Mud Density In g/cm3'].std()))&(df['Average Hookload kkgf']<(df['Average Hookload kkgf'].mean()+3*df['Average Hookload kkgf'].std()))&(df['Average Hookload kkgf']>(df['Average Hookload kkgf'].mean()-3*df['Average Hookload kkgf'].std()))&(df['USROP Gamma gAPI']<(df['USROP Gamma gAPI'].mean()+3*df['USROP Gamma gAPI'].std()))&(df['USROP Gamma gAPI']>(df['USROP Gamma gAPI'].mean()-3*df['USROP Gamma gAPI'].std()))]
df=df.reset_index(drop=True)

print(len(df))
print(df.head())

scaler=preprocessing.MinMaxScaler(feature_range=(0,1))


# In[102]:


'''
start_depth=df['Measured Depth m'][0]-1e-6
train_depth=start_depth+30
test_depth=train_depth+10

y_test_pred_all=[]

i=1

while True:
    if train_depth>df['Measured Depth m'][len(df)-1]:
        break
    if test_depth>df['Measured Depth m'][len(df)-1]:
        test_depth=df['Measured Depth m'][len(df)-1]
    
    df_train=df[(df['Measured Depth m']<=train_depth)&(df['Measured Depth m']>start_depth)]
    scaler.fit(df_train)
    df_train_scaled=scaler.transform(df_train)
    df_train_scaled=pd.DataFrame(df_train_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])
    
    x_train_scaled=df_train_scaled.drop(['Rate of Penetration m/h'],axis=1)
    y_train_scaled=df_train_scaled['Rate of Penetration m/h']
    
    rf=RandomForestRegressor(n_estimators=500)
    rf.fit(x_train_scaled,y_train_scaled)
    
    df_train_test=df[(df['Measured Depth m']<=test_depth)&(df['Measured Depth m']>start_depth)]
    test_scaled_start=(train_depth-df_train_test['Measured Depth m'].min())/(df_train_test['Measured Depth m'].max()-df_train_test['Measured Depth m'].min())
    scaler.fit(df_train_test)
    df_train_test_scaled=scaler.transform(df_train_test)
    df_train_test_scaled=pd.DataFrame(df_train_test_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])
    
    df_test_scaled=df_train_test_scaled[(df_train_test_scaled['Measured Depth m']>test_scaled_start)]
    x_test_scaled=df_test_scaled.drop(['Rate of Penetration m/h'],axis=1)
    y_test_scaled=df_test_scaled['Rate of Penetration m/h']
    
    y_test_pred_scaled=rf.predict(x_test_scaled)
    
    y_test_pred=y_test_pred_scaled*(max(df_train_test['Rate of Penetration m/h'])-min(df_train_test['Rate of Penetration m/h']))+min(df_train_test['Rate of Penetration m/h'])
    
    y_test_pred_all.append(y_test_pred)
    
    #start_depth=start_depth+10
    train_depth=train_depth+10
    test_depth=test_depth+10
    
    print('Iteration %d completed'%i)
    i=i+1
'''

X=df.drop(['Rate of Penetration m/h'],axis=1)
Y=df['Rate of Penetration m/h']

increment=577
train_depth=increment
test_depth=train_depth+increment

y_test_pred_all=[]

i=1

while True:
    if train_depth>df.index.max()+1:
        break
    if test_depth>df.index.max()+1:
        test_depth=df.index.max()+1
    
    x_train=X[0:train_depth]
    y_train=Y[0:train_depth]
    x_test=X[train_depth:test_depth]
    y_test=Y[train_depth:test_depth]
    
    #xgb=XGBRegressor()
    #xgb.fit(x_train,y_train)
    #y_test_pred=xgb.predict(x_test)
    
    rf=RandomForestRegressor(n_estimators=500)
    rf.fit(x_train,y_train)
    y_test_pred=rf.predict(x_test)
    
    y_test_pred_all.append(y_test_pred)
    
    train_depth=train_depth+increment
    test_depth=train_depth+increment
    
    print('Iteration %d completed'%i)
    i=i+1


# In[103]:


'''
all_test_data=df[(df['Measured Depth m']>(df['Measured Depth m'][0]-1e-6+30))]
y_test_all=np.array(all_test_data['Rate of Penetration m/h'])
print(y_test_all.shape)
print(y_test_all)

y_test_pred_all_array=np.concatenate(y_test_pred_all,axis=0)
print(y_test_pred_all_array.shape)
print(y_test_pred_all_array)

print(r2_score(y_test_all,y_test_pred_all_array))
print(mean_absolute_error(y_test_all,y_test_pred_all_array))
'''

y_test_pred_all_array=np.concatenate(y_test_pred_all,axis=0)

y_test_all=np.array(Y[577:])

print(r2_score(y_test_all,y_test_pred_all_array))
print(mean_absolute_error(y_test_all,y_test_pred_all_array))


# In[104]:


plt.scatter(y_test_all,y_test_pred_all_array)
plt.show()


# In[106]:


#x=df['Measured Depth m'][(df['Measured Depth m']>(start_depth+30))]

x=df['Measured Depth m'][577:]

plt.plot(x,y_test_pred_all_array,label='Prediction')
plt.plot(x,y_test_all,label='Actual')
plt.legend()
plt.show()


# In[ ]:





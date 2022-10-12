#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor

seed=1000
np.random.seed(seed)

scaler=preprocessing.MinMaxScaler(feature_range=(0,1))

df=pd.read_csv('USROP_A 4 N-SH_F-15Sd modified.csv')
df.drop(['Diameter mm','Hole Depth (TVD) m'],axis=1,inplace=True)

df.describe()


# In[2]:


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
    
    while len(df_train)==0:
        start_depth=start_depth+10
        train_depth=train_depth+10
        test_depth=test_depth+10
        df_train=df[(df['Measured Depth m']<=train_depth)&(df['Measured Depth m']>start_depth)]
    
    #scaler.fit(df_train)
    #df_train_scaled=scaler.transform(df_train)
    #df_train_scaled=pd.DataFrame(df_train_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])
    
    x_train=df_train.drop(['Rate of Penetration m/h'],axis=1)
    y_train=df_train['Rate of Penetration m/h']
    
    #x_train_scaled=df_train_scaled.drop(['Rate of Penetration m/h'],axis=1)
    #y_train_scaled=df_train_scaled['Rate of Penetration m/h']
    
    rf=RandomForestRegressor(n_estimators=500)
    rf.fit(x_train,y_train)
    #gb=GradientBoostingRegressor(n_estimators=1000)
    #gb.fit(x_train,y_train)
    #xgb=XGBRegressor(n_estimators=1000)
    #xgb.fit(x_train,y_train)
    
    #rf=RandomForestRegressor(n_estimators=500)
    #rf.fit(x_train_scaled,y_train_scaled)
    
    '''
    df_train_test=df[(df['Measured Depth m']<=test_depth)&(df['Measured Depth m']>start_depth)]
        
    test_scaled_start=(train_depth-df_train_test['Measured Depth m'].min())/(df_train_test['Measured Depth m'].max()-df_train_test['Measured Depth m'].min())
    scaler.fit(df_train_test)
    df_train_test_scaled=scaler.transform(df_train_test)
    df_train_test_scaled=pd.DataFrame(df_train_test_scaled,columns=['Measured Depth m','Weight on Bit kkgf','Average Standpipe Pressure kPa','Average Surface Torque kN.m','Rate of Penetration m/h','Average Rotary Speed rpm','Mud Flow In L/min','Mud Density In g/cm3','Average Hookload kkgf','USROP Gamma gAPI'])
    
    df_test_scaled=df_train_test_scaled[(df_train_test_scaled['Measured Depth m']>test_scaled_start)]
    
    while len(df_test_scaled)==0:
        start_depth=start_depth+10
        train_depth=train_depth+10
        test_depth=test_depth+10  
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
    '''
    
    
    df_test=df[(df['Measured Depth m']<=test_depth)&(df['Measured Depth m']>train_depth)]
    
    while len(df_test)==0:
        train_depth=train_depth+10
        test_depth=test_depth+10
        df_test=df[(df['Measured Depth m']<=test_depth)&(df['Measured Depth m']>train_depth)]
    
    x_test=df_test.drop(['Rate of Penetration m/h'],axis=1)
    y_test=df_test['Rate of Penetration m/h']
    
    y_test_pred=rf.predict(x_test)
    #y_test_pred=gb.predict(x_test)
    #y_test_pred=xgb.predict(x_test)
    
    y_test_pred_all.append(y_test_pred)
    
    start_depth=start_depth+10
    train_depth=train_depth+10
    test_depth=test_depth+10
    
    print('Iteration %d completed'%i)
    i=i+1


# In[7]:


all_test_data=df[(df['Measured Depth m']>(df['Measured Depth m'][0]-1e-6+30))]
y_test_all=np.array(all_test_data['Rate of Penetration m/h'])
print(y_test_all.shape)
print(y_test_all)

y_test_pred_all_array=np.concatenate(y_test_pred_all,axis=0)
print(y_test_pred_all_array.shape)
print(y_test_pred_all_array)

print(r2_score(y_test_all,y_test_pred_all_array))
print(mean_absolute_error(y_test_all,y_test_pred_all_array))

test_mape=np.sum(np.abs(y_test_all-y_test_pred_all_array)/y_test_all)*100/len(y_test_all)
print('MAPE for testing data is %f %%'%test_mape)


# In[4]:


plt.scatter(y_test_all,y_test_pred_all_array)
plt.show()


# In[5]:


y_test_dict={'Depth':all_test_data['Measured Depth m'],'True':y_test_all,'Prediction':y_test_pred_all_array}
y_test_df=pd.DataFrame(y_test_dict)
y_test_df=y_test_df[(y_test_df['True']>df['Rate of Penetration m/h'].mean()-3*df['Rate of Penetration m/h'].std())&(y_test_df['True']<df['Rate of Penetration m/h'].mean()+3*df['Rate of Penetration m/h'].std())]

print(r2_score(y_test_df['True'],y_test_df['Prediction']))
print(mean_absolute_error(y_test_df['True'],y_test_df['Prediction']))


# In[177]:


plt.scatter(y_test_df['True'],y_test_df['Prediction'])
plt.show()


# In[8]:


test_mape=np.sum(np.abs(y_test_df['True']-y_test_df['Prediction'])/y_test_df['True'])*100/len(y_test_df['True'])
print('MAPE for testing data is %f %%'%test_mape)


# In[9]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))


ax[0].plot(y_test_df['True'],y_test_df['Depth'])
ax[0].set_title('True ROP')
ax[0].set(xlabel='ROP (m/h)',ylabel='Depth (m)')
ax[0].grid(True)

ax[1].plot(y_test_df['Prediction'],y_test_df['Depth'])
ax[1].set_title('Predicted ROP')
ax[1].set(xlabel='ROP (m/h)',ylabel='Depth (m)')
ax[1].grid(True)

#plt.xlim([0,50])
plt.tight_layout()
plt.show()


# In[ ]:





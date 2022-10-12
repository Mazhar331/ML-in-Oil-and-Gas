#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import random as python_random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def reset_seeds():
    np.random.seed(100)
    python_random.seed(100)
    tf.random.set_seed(100)
reset_seeds()

df=pd.read_excel('Volve Well 11H all production data.xlsx')
df.dropna(how='any')
df=df.iloc[630:]
#df=df[(df['BORE_OIL_VOL']>=68)]

scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(df)
df_scaled=scaler.transform(df)
df_scaled=pd.DataFrame(df_scaled, columns=['DAYS','ON_STREAM_HRS','AVG_DOWNHOLE_PRESSURE','AVG_DOWNHOLE_TEMPERATURE','AVG_CHOKE_SIZE_P','AVG_WHP_P','AVG_WHT_P','BORE_OIL_VOL','BORE_GAS_VOL','BORE_WAT_VOL'])
#df_scaled.drop(['DAYS'],axis=1,inplace=True)

X=df_scaled
Y=df_scaled[['BORE_OIL_VOL','BORE_GAS_VOL','BORE_WAT_VOL']]

start_point=0

y_test_pred_all=[]

k=1

while True:
    if start_point+59>len(df):
        break
    
    current_set_x=X.iloc[start_point:start_point+59]
    current_set_y=Y.iloc[start_point:start_point+59]
    
    x_train=[]
    y_train=[]
    
    for i in range(15):
        x_train.append(current_set_x.iloc[i:i+30])
        y_train.append(current_set_y.iloc[i+30+15-1])
        
    x_test=[]
    
    for j in range(15,30): 
        x_test.append(current_set_x.iloc[j:j+30])
    
    x_train,y_train,x_test=np.array(x_train),np.array(y_train),np.array(x_test)
    
    forecast_LSTM=Sequential()
    forecast_LSTM.add(LSTM(units=30,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
    forecast_LSTM.add(Dense(units=3))
    opt=Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-07)
    forecast_LSTM.compile(optimizer=opt,loss='mean_squared_error')
    forecast_LSTM.fit(x_train,y_train,epochs=100,shuffle=True)
    
    y_test_pred=forecast_LSTM.predict(x_train)
    
    y_test_pred_all.append(y_test_pred)
    
    start_point=start_point+15
    
    print('Iteration %d completed'%k)
    print('\n')
    k=k+1


# In[3]:


y_test_all=Y.iloc[59:]
y_test_all=np.array(y_test_all)
print(y_test_all.shape)
print(y_test_all)

y_test_pred_all_array=np.concatenate(y_test_pred_all,axis=0)
y_test_pred_all_array=y_test_pred_all_array[:len(y_test_all)]
print(y_test_pred_all_array.shape)
print(y_test_pred_all_array)


# In[4]:


y_test_scaled_oil=y_test_all[:,0]
y_test_scaled_gas=y_test_all[:,1]
y_test_scaled_water=y_test_all[:,2]

y_test_pred_scaled_oil=y_test_pred_all_array[:,0]
y_test_pred_scaled_gas=y_test_pred_all_array[:,1]
y_test_pred_scaled_water=y_test_pred_all_array[:,2]


# In[5]:


y_test_oil=y_test_scaled_oil*(df['BORE_OIL_VOL'].max()-df['BORE_OIL_VOL'].min())+df['BORE_OIL_VOL'].min()
y_test_pred_oil=y_test_pred_scaled_oil*(df['BORE_OIL_VOL'].max()-df['BORE_OIL_VOL'].min())+df['BORE_OIL_VOL'].min()

y_test_gas=y_test_scaled_gas*(df['BORE_GAS_VOL'].max()-df['BORE_GAS_VOL'].min())+df['BORE_GAS_VOL'].min()
y_test_pred_gas=y_test_pred_scaled_gas*(df['BORE_GAS_VOL'].max()-df['BORE_GAS_VOL'].min())+df['BORE_GAS_VOL'].min()

y_test_water=y_test_scaled_water*(df['BORE_WAT_VOL'].max()-df['BORE_WAT_VOL'].min())+df['BORE_WAT_VOL'].min()
y_test_pred_water=y_test_pred_scaled_water*(df['BORE_WAT_VOL'].max()-df['BORE_WAT_VOL'].min())+df['BORE_WAT_VOL'].min()


# In[6]:


test_df=pd.DataFrame({'Oil True':y_test_oil,'Oil Prediction':y_test_pred_oil,'Gas True':y_test_gas,'Gas Prediction':y_test_pred_gas,'Water True':y_test_water,'Water Prediction':y_test_pred_water})
test_df.dropna(how='any')

test_df.describe()


# In[8]:


test_df=test_df[(test_df['Oil True']>=180)]

print('R2 for Oil is %f' %r2_score(test_df['Oil True'],test_df['Oil Prediction']))
print('MAE for Oil is %f' %mean_absolute_error(test_df['Oil True'],test_df['Oil Prediction']))

oil_mape=np.sum(np.abs(test_df['Oil True']-test_df['Oil Prediction'])/test_df['Oil True'])*100/len(test_df['Oil True'])
print('MAPE for Oil is %f %%'%oil_mape)

print('R2 for Gas is %f' %r2_score(test_df['Gas True'],test_df['Gas Prediction']))
print('MAE for Gas is %f' %mean_absolute_error(test_df['Gas True'],test_df['Gas Prediction']))

gas_mape=np.sum(np.abs(test_df['Gas True']-test_df['Gas Prediction'])/test_df['Gas True'])*100/len(test_df['Gas True'])
print('MAPE for Gas is %f %%'%gas_mape)

print('R2 for Water is %f' %r2_score(test_df['Water True'],test_df['Water Prediction']))
print('MAE for Water is %f' %mean_absolute_error(test_df['Water True'],test_df['Water Prediction']))

water_mape=np.sum(np.abs(test_df['Water True']-test_df['Water Prediction'])/test_df['Water True'])*100/len(test_df['Water True'])
print('MAPE for Water is %f %%'%water_mape)


# In[9]:


plt.plot(test_df['Oil True'])
plt.plot(test_df['Oil Prediction'])
plt.show()


# In[10]:


sns.distplot(test_df['Oil True']-test_df['Oil Prediction'])


# In[11]:


plt.plot(test_df['Gas True'])
plt.plot(test_df['Gas Prediction'])
plt.show()


# In[12]:


sns.distplot(test_df['Gas True']-test_df['Gas Prediction'])


# In[13]:


plt.plot(test_df['Water True'])
plt.plot(test_df['Water Prediction'])
plt.show()


# In[14]:


sns.distplot(test_df['Water True']-test_df['Water Prediction'])


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import random as python_random
import tensorflow as tf

def reset_seeds():
    np.random.seed(100)
    python_random.seed(100)
    tf.random.set_seed(100)
reset_seeds()

df=pd.read_excel('Volve Well 14H July 13-16 Production.xlsx')

scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(df)
df_scaled=scaler.transform(df)
df_scaled=pd.DataFrame(df_scaled, columns=['DAYS','ON_STREAM_HRS','AVG_DOWNHOLE_PRESSURE','AVG_DOWNHOLE_TEMPERATURE','AVG_CHOKE_SIZE_P','AVG_WHP_P','AVG_WHT_P','BORE_OIL_VOL','BORE_GAS_VOL','BORE_WAT_VOL'])

X=df_scaled
Y=df_scaled['BORE_OIL_VOL']

#x_blind=X[819:]
#y_blind=X[819:]
#x_model=X[:819]
#y_model=X[:819]

x_train=[]
y_train=[]

for i in range(60,573):
    x_train.append(X[i-60:i])
    y_train.append(Y[i])

x_train,y_train=np.array(x_train),np.array(y_train)

x_train.shape,y_train.shape


# In[2]:


forecast_LSTM=Sequential()
forecast_LSTM.add(LSTM(units=30, activation='tanh', input_shape=(x_train.shape[1],x_train.shape[2])))
forecast_LSTM.add(Dense(units=1))

forecast_LSTM.summary()


# In[3]:


opt=Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-07)

forecast_LSTM.compile(optimizer=opt,loss='mean_squared_error')

history=forecast_LSTM.fit(x_train,y_train,epochs=2000,batch_size=x_train.shape[0],shuffle=True)

y_train_pred=forecast_LSTM.predict(x_train)


# In[4]:


from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

y_train_unscaled=y_train*(max(df['BORE_OIL_VOL'])-min(df['BORE_OIL_VOL']))+min(df['BORE_OIL_VOL'])
y_train_pred_unscaled=y_train_pred*(max(df['BORE_OIL_VOL'])-min(df['BORE_OIL_VOL']))+min(df['BORE_OIL_VOL'])

print(y_train_unscaled.shape)
print(y_train_pred_unscaled.shape)

print('RMSE for training data is %f'%mean_squared_error(y_train,y_train_pred,squared=False))
print('R^2 for training data is %f'%r2_score(y_train_unscaled,y_train_pred_unscaled))

plt.scatter(y_train_unscaled,y_train_pred_unscaled)
plt.show()


# In[5]:


x_val=[]
y_val=[]

for i in range(573,696):
    x_val.append(X[i-60:i])
    y_val.append(Y[i])

x_val,y_val=np.array(x_val),np.array(y_val)

x_val.shape,y_val.shape


# In[6]:


y_val_pred=forecast_LSTM.predict(x_val)
print('R^2 for validation data is %f'%r2_score(y_val,y_val_pred))
plt.scatter(y_val,y_val_pred)
plt.show()


# In[7]:


plt.plot(y_val)


# In[ ]:





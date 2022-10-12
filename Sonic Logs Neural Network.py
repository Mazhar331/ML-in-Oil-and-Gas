#!/usr/bin/env python
# coding: utf-8

# In[29]:


import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

np.random.seed(1000)

scaler=preprocessing.MinMaxScaler(feature_range=(0,1))

well1=lasio.read('15_9-F-11A.LAS.txt')
well1_dict={'DEPTH':well1['DEPTH'],'NPHI':well1['NPHI'],'RHOB':well1['RHOB'],'GR':well1['GR'],'RT':well1['RT'],'PEF':well1['PEF'],'CALI':well1['CALI'],'DT':well1['DT'],'DTS':well1['DTS']}
well1_df=pd.DataFrame(well1_dict)
well1_df=well1_df[(well1_df['DEPTH']>=2600)&(well1_df['DEPTH']<=3720)]
well1_df.drop(['CALI'],axis=1,inplace=True)

well4=lasio.read('15_9-F-1B.LAS.txt')
well4_dict={'DEPTH':well4['DEPTH'],'NPHI':well4['NPHI'],'RHOB':well4['RHOB'],'GR':well4['GR'],'RT':well4['RT'],'PEF':well4['PEF'],'CALI':well4['CALI'],'DT':well4['DT'],'DTS':well4['DTS']}
well4_df=pd.DataFrame(well4_dict)
well4_df=well4_df[(well4_df['DEPTH']>=3100)&(well4_df['DEPTH']<=3400)]
well4_df.drop(['CALI'],axis=1,inplace=True)

well1_df['log GR']=np.log10(well1_df['GR'])
well1_df['log RT']=np.log10(well1_df['RT'])
well1_df.drop(['GR','RT'],axis=1,inplace=True)

well4_df['log GR']=np.log10(well4_df['GR'])
well4_df['log RT']=np.log10(well4_df['RT'])
well4_df.drop(['GR','RT'],axis=1,inplace=True)

well1_df=well1_df[(well1_df['NPHI']<well1_df['NPHI'].mean()+3*well1_df['NPHI'].std())&(well1_df['NPHI']>well1_df['NPHI'].mean()-3*well1_df['NPHI'].std())&(well1_df['RHOB']<well1_df['RHOB'].mean()+3*well1_df['RHOB'].std())&(well1_df['RHOB']>well1_df['RHOB'].mean()-3*well1_df['RHOB'].std())&(well1_df['log GR']<well1_df['log GR'].mean()+3*well1_df['log GR'].std())&(well1_df['log GR']>well1_df['log GR'].mean()-3*well1_df['log GR'].std())&(well1_df['log RT']<well1_df['log RT'].mean()+3*well1_df['log RT'].std())&(well1_df['log RT']>well1_df['log RT'].mean()-3*well1_df['log RT'].std())&(well1_df['PEF']<well1_df['PEF'].mean()+3*well1_df['PEF'].std())&(well1_df['PEF']>well1_df['PEF'].mean()-3*well1_df['PEF'].std())&(well1_df['DT']<well1_df['DT'].mean()+3*well1_df['DT'].std())&(well1_df['DT']>well1_df['DT'].mean()-3*well1_df['DT'].std())&(well1_df['DTS']<well1_df['DTS'].mean()+3*well1_df['DTS'].std())&(well1_df['DTS']>well1_df['DTS'].mean()-3*well1_df['DTS'].std())]

well4_df=well4_df[(well4_df['NPHI']<well4_df['NPHI'].mean()+3*well4_df['NPHI'].std())&(well4_df['NPHI']>well4_df['NPHI'].mean()-3*well4_df['NPHI'].std())&(well4_df['RHOB']<well4_df['RHOB'].mean()+3*well4_df['RHOB'].std())&(well4_df['RHOB']>well4_df['RHOB'].mean()-3*well4_df['RHOB'].std())&(well4_df['log GR']<well4_df['log GR'].mean()+3*well4_df['log GR'].std())&(well4_df['log GR']>well4_df['log GR'].mean()-3*well4_df['log GR'].std())&(well4_df['log RT']<well4_df['log RT'].mean()+3*well4_df['log RT'].std())&(well4_df['log RT']>well4_df['log RT'].mean()-3*well4_df['log RT'].std())&(well4_df['PEF']<well4_df['PEF'].mean()+3*well4_df['PEF'].std())&(well4_df['PEF']>well4_df['PEF'].mean()-3*well4_df['PEF'].std())&(well4_df['DT']<well4_df['DT'].mean()+3*well4_df['DT'].std())&(well4_df['DT']>well4_df['DT'].mean()-3*well4_df['DT'].std())&(well4_df['DTS']<well4_df['DTS'].mean()+3*well4_df['DTS'].std())&(well4_df['DTS']>well4_df['DTS'].mean()-3*well4_df['DTS'].std())]

training_data=pd.concat([well1_df,well4_df])
print('Training data size: %d'%len(training_data))

training_data.describe()


# In[30]:


scaler.fit(training_data)
training_data_scaled=scaler.transform(training_data)
training_data_scaled=pd.DataFrame(training_data_scaled,columns=['DEPTH','NPHI','RHOB','PEF','DT','DTS','log GR','log RT'])

DT_train_scaled=training_data_scaled['DT']
DTS_train_scaled=training_data_scaled['DTS']
x_train_scaled=training_data_scaled.drop(['DT','DTS'],axis=1)

well3=lasio.read('15_9-F-1A.LAS.txt')
well3_dict={'DEPTH':well3['DEPTH'],'NPHI':well3['NPHI'],'RHOB':well3['RHOB'],'GR':well3['GR'],'RT':well3['RT'],'PEF':well3['PEF'],'CALI':well3['CALI'],'DT':well3['DT'],'DTS':well3['DTS']}
well3_df=pd.DataFrame(well3_dict)
well3_df=well3_df[(well3_df['DEPTH']>=2620)&(well3_df['DEPTH']<=3640)]
well3_df.drop(['CALI'],axis=1,inplace=True)

well3_df['log GR']=np.log10(well3_df['GR'])
well3_df['log RT']=np.log10(well3_df['RT'])
well3_df.drop(['GR','RT'],axis=1,inplace=True)

well3_df=well3_df[(well3_df['NPHI']<well3_df['NPHI'].mean()+3*well3_df['NPHI'].std())&(well3_df['NPHI']>well3_df['NPHI'].mean()-3*well3_df['NPHI'].std())&(well3_df['RHOB']<well3_df['RHOB'].mean()+3*well3_df['RHOB'].std())&(well3_df['RHOB']>well3_df['RHOB'].mean()-3*well3_df['RHOB'].std())&(well3_df['log GR']<well3_df['log GR'].mean()+3*well3_df['log GR'].std())&(well3_df['log GR']>well3_df['log GR'].mean()-3*well3_df['log GR'].std())&(well3_df['log RT']<well3_df['log RT'].mean()+3*well3_df['log RT'].std())&(well3_df['log RT']>well3_df['log RT'].mean()-3*well3_df['log RT'].std())&(well3_df['PEF']<well3_df['PEF'].mean()+3*well3_df['PEF'].std())&(well3_df['PEF']>well3_df['PEF'].mean()-3*well3_df['PEF'].std())&(well3_df['DT']<well3_df['DT'].mean()+3*well3_df['DT'].std())&(well3_df['DT']>well3_df['DT'].mean()-3*well3_df['DT'].std())&(well3_df['DTS']<well3_df['DTS'].mean()+3*well3_df['DTS'].std())&(well3_df['DTS']>well3_df['DTS'].mean()-3*well3_df['DTS'].std())]
well3_df=well3_df[(well3_df['NPHI']<=training_data['NPHI'].max())&(well3_df['NPHI']>=training_data['NPHI'].min())&(well3_df['RHOB']<=training_data['RHOB'].max())&(well3_df['RHOB']>=training_data['RHOB'].min())&(well3_df['log GR']<=training_data['log GR'].max())&(well3_df['log GR']>=training_data['log GR'].min())&(well3_df['log RT']<=training_data['log RT'].max())&(well3_df['log RT']>=training_data['log RT'].min())&(well3_df['PEF']<=training_data['PEF'].max())&(well3_df['PEF']>=training_data['PEF'].min())&(well3_df['DT']<=training_data['DT'].max())&(well3_df['DT']>=training_data['DT'].min())&(well3_df['DTS']<=training_data['DTS'].max())&(well3_df['DTS']>=training_data['DTS'].min())]

well3_df_scaled=scaler.transform(well3_df)
well3_df_scaled=pd.DataFrame(well3_df_scaled,columns=['DEPTH','NPHI','RHOB','PEF','DT','DTS','log GR','log RT'])

DT_test_scaled=well3_df_scaled['DT']
DTS_test_scaled=well3_df_scaled['DTS']
x_test_scaled=well3_df_scaled.drop(['DT','DTS'],axis=1)


# In[32]:


DT_model=Sequential()
DT_model.add(Dense(24,activation='sigmoid',input_dim=6))
DT_model.add(Dense(24,activation='sigmoid'))
DT_model.add(Dense(12,activation='sigmoid'))
DT_model.add(Dense(12,activation='sigmoid'))
DT_model.add(Dense(6,activation='sigmoid'))
DT_model.add(Dense(1))
DT_model.compile(optimizer='adam',loss='mean_squared_error')
DT_model.fit(x_train_scaled,DT_train_scaled,epochs=100,validation_data=(x_test_scaled,DT_test_scaled))
DT_train_scaled_pred=DT_model.predict(x_train_scaled)

DTS_model=Sequential()
DTS_model.add(Dense(24,activation='sigmoid',input_dim=6))
DTS_model.add(Dense(24,activation='sigmoid'))
DTS_model.add(Dense(12,activation='sigmoid'))
DTS_model.add(Dense(12,activation='sigmoid'))
DTS_model.add(Dense(6,activation='sigmoid'))
DTS_model.add(Dense(1))
DTS_model.compile(optimizer='adam',loss='mean_squared_error')
DTS_model.fit(x_train_scaled,DTS_train_scaled,epochs=100,validation_data=(x_test_scaled,DTS_test_scaled))
DTS_train_scaled_pred=DTS_model.predict(x_train_scaled)

DT_train_pred=DT_train_scaled_pred*(training_data['DT'].max()-training_data['DT'].min())+training_data['DT'].min()
DTS_train_pred=DTS_train_scaled_pred*(training_data['DTS'].max()-training_data['DTS'].min())+training_data['DTS'].min()

print('R^2 for DT training is %f'%r2_score(training_data['DT'],DT_train_pred))
print('R^2 for DTS training is %f'%r2_score(training_data['DTS'],DTS_train_pred))
print('RMSE for DT training is %f'%mean_squared_error(training_data['DT'],DT_train_pred,squared=False))
print('RMSE for DTS training is %f'%mean_squared_error(training_data['DTS'],DTS_train_pred,squared=False))

DT_test_scaled_pred=DT_model.predict(x_test_scaled)
DTS_test_scaled_pred=DTS_model.predict(x_test_scaled)

DT_test_pred=DT_test_scaled_pred*(training_data['DT'].max()-training_data['DT'].min())+training_data['DT'].min()
DTS_test_pred=DTS_test_scaled_pred*(training_data['DTS'].max()-training_data['DTS'].min())+training_data['DTS'].min()

print('R^2 for DT testing is %f'%r2_score(DT_test_scaled,DT_test_scaled_pred))
print('R^2 for DTS testing is %f'%r2_score(DTS_test_scaled,DTS_test_scaled_pred))
print('RMSE for DT testing is %f'%mean_squared_error(well3_df['DT'],DT_test_pred,squared=False))
print('RMSE for DTS testing is %f'%mean_squared_error(well3_df['DTS'],DTS_test_pred,squared=False))


# In[35]:


x_train_scaled2=training_data_scaled.drop(['DTS'],axis=1)

DTS_model2=Sequential()
DTS_model2.add(Dense(28,activation='sigmoid',input_dim=7))
DTS_model2.add(Dense(28,activation='sigmoid'))
DTS_model2.add(Dense(14,activation='sigmoid'))
DTS_model2.add(Dense(14,activation='sigmoid'))
DTS_model2.add(Dense(7,activation='sigmoid'))
DTS_model2.add(Dense(1))
DTS_model2.compile(optimizer='adam',loss='mean_squared_error')
DTS_model2.fit(x_train_scaled2,DTS_train_scaled,epochs=100)
DTS_train_scaled_pred2=DTS_model2.predict(x_train_scaled2)

DTS_train_pred2=DTS_train_scaled_pred2*(training_data['DTS'].max()-training_data['DTS'].min())+training_data['DTS'].min()

print('R^2 for DTS training including DT is %f'%r2_score(training_data['DTS'],DTS_train_pred2))
print('RMSE for DTS training including DT is %f'%mean_squared_error(training_data['DTS'],DTS_train_pred2,squared=False))


# In[36]:


x_test_scaled2=well3_df_scaled.drop(['DTS'],axis=1)
x_test_scaled2['DT']=DT_test_scaled_pred

DTS_test_scaled_pred2=DTS_model2.predict(x_test_scaled2)

DTS_test_pred2=DTS_test_scaled_pred2*(training_data['DTS'].max()-training_data['DTS'].min())+training_data['DTS'].min()

print('R^2 for DTS testing including DT is %f'%r2_score(well3_df['DTS'],DTS_test_pred2))
print('RMSE for DTS testing including DT is %f'%mean_squared_error(well3_df['DTS'],DTS_test_pred2,squared=False))


# In[ ]:





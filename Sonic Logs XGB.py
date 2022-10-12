#!/usr/bin/env python
# coding: utf-8

# In[7]:


import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import seaborn as sns

np.random.seed(1000)

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

DT_train=training_data['DT']
DTS_train=training_data['DTS']
x_train=training_data.drop(['DT','DTS'],axis=1)

xgb1=XGBRegressor()
xgb1.fit(x_train,DT_train)
DT_train_pred=xgb1.predict(x_train)

xgb2=XGBRegressor()
xgb2.fit(x_train,DTS_train)
DTS_train_pred=xgb2.predict(x_train)

print('R^2 for DT training is %f'%r2_score(DT_train,DT_train_pred))
print('R^2 for DTS training is %f'%r2_score(DTS_train,DTS_train_pred))
print('RMSE for DT training is %f'%mean_squared_error(DT_train,DT_train_pred,squared=False))
print('RMSE for DTS training is %f'%mean_squared_error(DTS_train,DTS_train_pred,squared=False))


# In[8]:


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

DT_test=well3_df['DT']
DTS_test=well3_df['DTS']
x_test=well3_df.drop(['DT','DTS'],axis=1)

DT_test_pred=xgb1.predict(x_test)
DTS_test_pred=xgb2.predict(x_test)

print('R^2 for DT testing is %f'%r2_score(DT_test,DT_test_pred))
print('R^2 for DTS testing is %f'%r2_score(DTS_test,DTS_test_pred))
print('RMSE for DT testing is %f'%mean_squared_error(DT_test,DT_test_pred,squared=False))
print('RMSE for DTS testing is %f'%mean_squared_error(DTS_test,DTS_test_pred,squared=False))


# In[9]:


shear_train=training_data.drop(['DTS'],axis=1)

xgb3=XGBRegressor()
xgb3.fit(shear_train,DTS_train)
shear_train_pred=xgb3.predict(shear_train)

print('R^2 for DTS training data including DT is %f' %r2_score(DTS_train,shear_train_pred))
print('RMSE for DTS training data including DT is %f' %mean_squared_error(DTS_train,shear_train_pred,squared=False))


# In[10]:


shear_test=well3_df.drop(['DTS'],axis=1)
shear_test['DT']=DT_test_pred

shear_test_pred=xgb3.predict(shear_test)

print('R^2 for DTS testing data including DT is %f' %r2_score(DTS_test,shear_test_pred))
print('RMSE for DTS testing data including DT is %f' %mean_squared_error(DTS_test,shear_test_pred,squared=False))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import seaborn as sns

well1=lasio.read('15_9-F-11A.LAS.txt')
well1_dict={'DEPTH':well1['DEPTH'],'NPHI':well1['NPHI'],'RHOB':well1['RHOB'],'GR':well1['GR'],'RT':well1['RT'],'PEF':well1['PEF'],'CALI':well1['CALI'],'DT':well1['DT'],'DTS':well1['DTS']}
well1_df=pd.DataFrame(well1_dict)
well1_df=well1_df[(well1_df['DEPTH']>=2600)&(well1_df['DEPTH']<=3720)]

print(well1_df.describe())

sns.pairplot(well1_df,diag_kind='kde',plot_kws={'alpha':0.6,'s':30,'edgecolor':'k'})
plt.show()


# In[2]:


well4=lasio.read('15_9-F-1B.LAS.txt')
well4_dict={'DEPTH':well4['DEPTH'],'NPHI':well4['NPHI'],'RHOB':well4['RHOB'],'GR':well4['GR'],'RT':well4['RT'],'PEF':well4['PEF'],'CALI':well4['CALI'],'DT':well4['DT'],'DTS':well4['DTS']}
well4_df=pd.DataFrame(well4_dict)
well4_df=well4_df[(well4_df['DEPTH']>=3100)&(well4_df['DEPTH']<=3400)]

print(well4_df.describe())

sns.pairplot(well4_df,diag_kind='kde',plot_kws={'alpha':0.6,'s':30,'edgecolor':'k'})
plt.show()


# In[3]:


well1_df['log GR']=np.log10(well1_df['GR'])
well1_df['log RT']=np.log10(well1_df['RT'])
well1_df.drop(['GR','RT'],axis=1,inplace=True)

well4_df['log GR']=np.log10(well4_df['GR'])
well4_df['log RT']=np.log10(well4_df['RT'])
well4_df.drop(['GR','RT'],axis=1,inplace=True)


# In[4]:


well1_df=well1_df[(well1_df['NPHI']<well1_df['NPHI'].mean()+3*well1_df['NPHI'].std())&(well1_df['NPHI']>well1_df['NPHI'].mean()-3*well1_df['NPHI'].std())&(well1_df['RHOB']<well1_df['RHOB'].mean()+3*well1_df['RHOB'].std())&(well1_df['RHOB']>well1_df['RHOB'].mean()-3*well1_df['RHOB'].std())&(well1_df['log GR']<well1_df['log GR'].mean()+3*well1_df['log GR'].std())&(well1_df['log GR']>well1_df['log GR'].mean()-3*well1_df['log GR'].std())&(well1_df['log RT']<well1_df['log RT'].mean()+3*well1_df['log RT'].std())&(well1_df['log RT']>well1_df['log RT'].mean()-3*well1_df['log RT'].std())&(well1_df['PEF']<well1_df['PEF'].mean()+3*well1_df['PEF'].std())&(well1_df['PEF']>well1_df['PEF'].mean()-3*well1_df['PEF'].std())&(well1_df['CALI']<well1_df['CALI'].mean()+3*well1_df['CALI'].std())&(well1_df['CALI']>well1_df['CALI'].mean()-3*well1_df['CALI'].std())&(well1_df['DT']<well1_df['DT'].mean()+3*well1_df['DT'].std())&(well1_df['DT']>well1_df['DT'].mean()-3*well1_df['DT'].std())&(well1_df['DTS']<well1_df['DTS'].mean()+3*well1_df['DTS'].std())&(well1_df['DTS']>well1_df['DTS'].mean()-3*well1_df['DTS'].std())]

well4_df=well4_df[(well4_df['NPHI']<well4_df['NPHI'].mean()+3*well4_df['NPHI'].std())&(well4_df['NPHI']>well4_df['NPHI'].mean()-3*well4_df['NPHI'].std())&(well4_df['RHOB']<well4_df['RHOB'].mean()+3*well4_df['RHOB'].std())&(well4_df['RHOB']>well4_df['RHOB'].mean()-3*well4_df['RHOB'].std())&(well4_df['log GR']<well4_df['log GR'].mean()+3*well4_df['log GR'].std())&(well4_df['log GR']>well4_df['log GR'].mean()-3*well4_df['log GR'].std())&(well4_df['log RT']<well4_df['log RT'].mean()+3*well4_df['log RT'].std())&(well4_df['log RT']>well4_df['log RT'].mean()-3*well4_df['log RT'].std())&(well4_df['PEF']<well4_df['PEF'].mean()+3*well4_df['PEF'].std())&(well4_df['PEF']>well4_df['PEF'].mean()-3*well4_df['PEF'].std())&(well4_df['CALI']<well4_df['CALI'].mean()+3*well4_df['CALI'].std())&(well4_df['CALI']>well4_df['CALI'].mean()-3*well4_df['CALI'].std())&(well4_df['DT']<well4_df['DT'].mean()+3*well4_df['DT'].std())&(well4_df['DT']>well4_df['DT'].mean()-3*well4_df['DT'].std())&(well4_df['DTS']<well4_df['DTS'].mean()+3*well4_df['DTS'].std())&(well4_df['DTS']>well4_df['DTS'].mean()-3*well4_df['DTS'].std())]

training_data=pd.concat([well1_df,well4_df])
print('Training data size: %d'%len(training_data))

sns.pairplot(training_data,diag_kind='kde',plot_kws={'alpha':0.6,'s':30,'edgecolor':'k'})
plt.show()

plt.figure(figsize=(14,10))
sns.heatmap(training_data.corr(),annot=True)
plt.title('Feature Corrletaion Matrix')
plt.show()


# In[5]:


y_train=training_data[['DT','DTS']]
x_train=training_data.drop(['DT','DTS'],axis=1)

rf1=RandomForestRegressor(random_state=1000)
RF=MultiOutputRegressor(rf1)
RF.fit(x_train,y_train)
y_train_pred=RF.predict(x_train)

print('R^2 for DT training data is %f' %r2_score(y_train['DT'],y_train_pred[:,0]))
print('R^2 for DTS training data is %f' %r2_score(y_train['DTS'],y_train_pred[:,1]))


# In[6]:


print('RMSE for DT training data is %f' %mean_squared_error(y_train['DT'],y_train_pred[:,0],squared=False))
print('RMSE for DTS training data is %f' %mean_squared_error(y_train['DTS'],y_train_pred[:,1],squared=False))


# In[35]:


well3=lasio.read('15_9-F-1A.LAS.txt')
well3_dict={'DEPTH':well3['DEPTH'],'NPHI':well3['NPHI'],'RHOB':well3['RHOB'],'GR':well3['GR'],'RT':well3['RT'],'PEF':well3['PEF'],'CALI':well3['CALI'],'DT':well3['DT'],'DTS':well3['DTS']}
well3_df=pd.DataFrame(well3_dict)
well3_df=well3_df[(well3_df['DEPTH']>=2620)&(well3_df['DEPTH']<=3640)]

well3_df['log GR']=np.log10(well3_df['GR'])
well3_df['log RT']=np.log10(well3_df['RT'])
well3_df.drop(['GR','RT'],axis=1,inplace=True)

well3_df=well3_df[(well3_df['NPHI']<well3_df['NPHI'].mean()+3*well3_df['NPHI'].std())&(well3_df['NPHI']>well3_df['NPHI'].mean()-3*well3_df['NPHI'].std())&(well3_df['RHOB']<well3_df['RHOB'].mean()+3*well3_df['RHOB'].std())&(well3_df['RHOB']>well3_df['RHOB'].mean()-3*well3_df['RHOB'].std())&(well3_df['log GR']<well3_df['log GR'].mean()+3*well3_df['log GR'].std())&(well3_df['log GR']>well3_df['log GR'].mean()-3*well3_df['log GR'].std())&(well3_df['log RT']<well3_df['log RT'].mean()+3*well3_df['log RT'].std())&(well3_df['log RT']>well3_df['log RT'].mean()-3*well3_df['log RT'].std())&(well3_df['PEF']<well3_df['PEF'].mean()+3*well3_df['PEF'].std())&(well3_df['PEF']>well3_df['PEF'].mean()-3*well3_df['PEF'].std())&(well3_df['CALI']<well3_df['CALI'].mean()+3*well3_df['CALI'].std())&(well3_df['CALI']>well3_df['CALI'].mean()-3*well3_df['CALI'].std())&(well3_df['DT']<well3_df['DT'].mean()+3*well3_df['DT'].std())&(well3_df['DT']>well3_df['DT'].mean()-3*well3_df['DT'].std())&(well3_df['DTS']<well3_df['DTS'].mean()+3*well3_df['DTS'].std())&(well3_df['DTS']>well3_df['DTS'].mean()-3*well3_df['DTS'].std())]
well3_df=well3_df[(well3_df['NPHI']<=training_data['NPHI'].max())&(well3_df['NPHI']>=training_data['NPHI'].min())&(well3_df['RHOB']<=training_data['RHOB'].max())&(well3_df['RHOB']>=training_data['RHOB'].min())&(well3_df['log GR']<=training_data['log GR'].max())&(well3_df['log GR']>=training_data['log GR'].min())&(well3_df['log RT']<=training_data['log RT'].max())&(well3_df['log RT']>=training_data['log RT'].min())&(well3_df['PEF']<=training_data['PEF'].max())&(well3_df['PEF']>=training_data['PEF'].min())&(well3_df['CALI']<=training_data['CALI'].max())&(well3_df['CALI']>=training_data['CALI'].min())&(well3_df['DT']<=training_data['DT'].max())&(well3_df['DT']>=training_data['DT'].min())&(well3_df['DTS']<=training_data['DTS'].max())&(well3_df['DTS']>=training_data['DTS'].min())]

sns.pairplot(well3_df,diag_kind='kde',plot_kws={'alpha':0.6,'s':30,'edgecolor':'k'})
plt.show()

plt.figure(figsize=(14,10))
sns.heatmap(well3_df.corr(),annot=True)
plt.title('Feature Corrletaion Matrix')
plt.show()


# In[36]:


y_test=well3_df[['DT','DTS']]
x_test=well3_df.drop(['DT','DTS'],axis=1)

y_test_pred=RF.predict(x_test)

print('R^2 for DT testing data is %f' %r2_score(y_test['DT'],y_test_pred[:,0]))
print('R^2 for DTS testing data is %f' %r2_score(y_test['DTS'],y_test_pred[:,1]))
print(y_test_pred[:,0])


# In[37]:


print('RMSE for DT testing data is %f' %mean_squared_error(y_test['DT'],y_test_pred[:,0],squared=False))
print('RMSE for DTS testing data is %f' %mean_squared_error(y_test['DTS'],y_test_pred[:,1],squared=False))


# In[38]:


plt.scatter(y_test['DT'],y_test_pred[:,0])
plt.show()


# In[39]:


plt.scatter(y_test['DTS'],y_test_pred[:,1])
plt.show()


# In[107]:


'''
DT_train_R2=[]
DT_test_R2=[]
DTS_train_R2=[]
DTS_test_R2=[]
DT_train_RMSE=[]
DT_test_RMSE=[]
DTS_train_RMSE=[]
DTS_test_RMSE=[]

i=0

for n in range(50,1000,10):
    rf1=RandomForestRegressor(n_estimators=n,random_state=1000)
    RF=MultiOutputRegressor(rf1)
    RF.fit(x_train,y_train)
    y_train_pred=RF.predict(x_train)

    DT_train_R2.append(r2_score(y_train['DT'],y_train_pred[:,0]))
    DTS_train_R2.append(r2_score(y_train['DTS'],y_train_pred[:,1]))
    DT_train_RMSE.append(mean_squared_error(y_train['DT'],y_train_pred[:,0],squared=False))
    DTS_train_RMSE.append(mean_squared_error(y_train['DTS'],y_train_pred[:,1],squared=False))
    
    y_test_pred=RF.predict(x_test)

    DT_test_R2.append(r2_score(y_test['DT'],y_test_pred[:,0]))
    DTS_test_R2.append(r2_score(y_test['DTS'],y_test_pred[:,1]))
    DT_test_RMSE.append(mean_squared_error(y_test['DT'],y_test_pred[:,0],squared=False))
    DTS_test_RMSE.append(mean_squared_error(y_test['DTS'],y_test_pred[:,1],squared=False))
    
    i=i+1
    print('Iteration %d completed'%i)
'''


# In[108]:


'''
plt.plot(np.arange(50,1000,10),DT_train_R2,label='DT_train_R2')
plt.plot(np.arange(50,1000,10),DT_test_R2,label='DT_test_R2')
plt.plot(np.arange(50,1000,10),DTS_train_R2,label='DTS_train_R2')
plt.plot(np.arange(50,1000,10),DTS_test_R2,label='DTS_test_R2')
plt.legend()
plt.show()
'''


# In[110]:


'''
plt.plot(np.arange(50,1000,10),DT_train_RMSE,label='DT_train_RMSE')
plt.plot(np.arange(50,1000,10),DT_test_RMSE,label='DT_test_RMSE')
plt.plot(np.arange(50,1000,10),DTS_train_RMSE,label='DTS_train_RMSE')
plt.plot(np.arange(50,1000,10),DTS_test_RMSE,label='DTS_test_RMSE')
plt.legend()
plt.show()
'''


# In[40]:


shear_y_train=training_data['DTS']
shear_x_train=training_data.drop(['DTS'],axis=1)

rf2=RandomForestRegressor(random_state=1000)
rf2.fit(shear_x_train,shear_y_train)
shear_y_train_pred=rf2.predict(shear_x_train)

print('R^2 for DTS training data including DT is %f' %r2_score(shear_y_train,shear_y_train_pred))
print('RMSE for DTS training data including DT is %f' %mean_squared_error(shear_y_train,shear_y_train_pred,squared=False))


# In[41]:


shear_y_test=well3_df['DTS']
shear_x_test=well3_df.drop(['DTS'],axis=1)
shear_x_test['DT']=y_test_pred[:,0]
print(shear_x_test.head())

shear_y_test_pred=rf2.predict(shear_x_test)

print('R^2 for DTS testing data including DT is %f' %r2_score(shear_y_test,shear_y_test_pred))
print('RMSE for DTS testing data including DT is %f' %mean_squared_error(shear_y_test,shear_y_test_pred,squared=False))


# In[42]:


plt.scatter(shear_y_test,shear_y_test_pred)
plt.show()


# In[ ]:





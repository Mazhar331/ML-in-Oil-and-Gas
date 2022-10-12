#!/usr/bin/env python
# coding: utf-8

# In[9]:


import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

well1=lasio.read('15_9-F-11A.LAS.txt') #train

logs = ['NPHI', 'RHOB', 'GR', 'RT', 'PEF', 'CALI', 'DT','DTS']

# create the subplots; ncols equals the number of logs
fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=(20,10))

# looping each log to display in the subplots

colors = ['black', 'red', 'blue', 'green', 'purple', 'black', 'orange','yellow']

for i in range(len(logs)):
    if i == 3:
    # for resistivity, semilog plot
        ax[i].semilogx(well1[logs[i]], well1['DEPTH'], color=colors[i])
    else:
    # for non-resistivity, normal plot
        ax[i].plot(well1[logs[i]], well1['DEPTH'], color=colors[i])
  
    ax[i].set_title(logs[i])
    ax[i].grid(True)

ax[2].set_xlim(0, 300)
plt.tight_layout(1.1)
plt.show()


# In[10]:


print(well1['DEPTH'])

well1=pd.DataFrame(well1)
well1=well1[(well1['DEPTH']>=2600)&(well1['DEPTH']<=3720)]

'''
well3=lasio.read('15_9-F-1A.LAS.txt') #test
well4=lasio.read('15_9-F-1B.LAS.txt') #train

well1=well1.df()
well1=well1.loc[(well1['DEPTH']>=2600)&(well1['DEPTH']<=3720)]

well3=well3.df()
well3=well3[(well3['DEPTH']>=2620)&(well3['DEPTH']<=3640)]

well4=well4.df()
well4=well4[(well4['DEPTH']>=3100)&(well4['DEPTH']<=3400)]
'''


# In[ ]:





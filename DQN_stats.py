#!/usr/bin/env python
# coding: utf-8

# In[12]:



# In[13]:


import pandas as pd
stats = pd.read_csv('log/dqn.out', header=None)
stats.head()


# In[14]:


def stripScore(x):
    return float(x.strip().split(' : ')[1])
scores = stats[2].apply(lambda x: stripScore(x))
scores.head()


# In[15]:


scores.plot()


# In[ ]:





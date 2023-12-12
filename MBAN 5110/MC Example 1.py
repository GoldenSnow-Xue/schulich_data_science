#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


values = np.array([1,2,3,4,5,6,7,8])
hits = np.array([12,13,43,65,10,5,4,3])


# In[5]:


# Now we calculate the probability values from hits (i.e., frequencies)
probs = hits/np.sum(hits)
probs


# In[6]:


cum_probs = np.cumsum(probs)
cum_probs


# In[29]:


# we now generate a random variable between zero and one
random_variable = np.random.uniform(0,1,1)
# this function gives the index value of the row such that random variable lies between lower and upper bounds
index_value = min(np.argwhere(cum_probs>random_variable).ravel())
# the assigned value then becomes
assigned_value = values[index_value]


# In[31]:


# we now calculate the profit of the agent
profit = 3*(assigned_value<5)+5*(assigned_value>=5)


# In[36]:


list([random_variable.ravel(), assigned_value, profit])


# In[78]:


m = 1000
N = 100
random_variable = np.random.uniform(0,1,[m,N])
profit_values = np.zeros((m,N))
for i in range(N):
    profit_m = []
    for j in range(m):
        index_value = min(np.argwhere(cum_probs>random_variable[j,i]).ravel())
        assigned_value = values[index_value]
        profit_val = 3*(assigned_value<5)+5*(assigned_value>=5)
        profit_m.append(profit_val)
    profit_values[:,i] = profit_m
    


# In[81]:


# calculate the mean of each sample
mean_sample = np.mean(profit_values, axis = 0)
# calculate standard deviation
sigma = np.std(mean_sample)
# Mean of means
f_bar_bar = np.mean(mean_sample)
# Lower bound of CI
lower_bound = f_bar_bar-1.96*sigma/np.sqrt(N)
# Upper bound of CI
upper_bound = f_bar_bar+1.96*sigma/np.sqrt(N)
# Summary statistics Lower Bound, Mean of Means, Upper Bound
list[lower_bound,f_bar_bar,upper_bound]


# In[ ]:





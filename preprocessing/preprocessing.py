#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ## Data loading

# In[2]:


data = pd.read_csv("../data/SWOW-EN.complete.csv")


# In[3]:


data


# We want to check if there are some participants with few entries. These should be filtered out

# In[4]:


participant_frequency = data["participantID"].value_counts()


# In[5]:


print(min(participant_frequency), max(participant_frequency))


# We're good on that, so now we need to take a subset of the data. This ensures that our classes (countries) are balanced and also makes the data more manageable.

# First, let's filter out entries with unknown words

# In[6]:


data = data[data["R1Raw"] != "Unknown word"]


# We only want to predict the top 10 most frequent countries, taking an equal amount of data from each.

# In[7]:


countries = list(data["country"].value_counts()[:10].index)
countries


# In[8]:


filtered = pd.concat(list(map(lambda country: data[data["country"] == country].sample(10000), countries)))


# In[9]:


filtered = filtered[["country", "age", "gender", "cue", "R1Raw", "R2Raw", "R3Raw"]]


# We need to check how many entries a participant has made

# In[10]:


filtered["amount"] = filtered.apply(lambda row: 1 if row["R2Raw"] == "No more responses" else 2 if row["R3Raw"] == "No more responses" else 3, axis=1)


# In[12]:


filtered.to_csv("SWOW-EN.complete_preprocessed.csv")


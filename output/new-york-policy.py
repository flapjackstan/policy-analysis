#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import nltk
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

#C:\Users\elmsc\AppData\Roaming\nltk_data
#https://stackoverflow.com/questions/40206249/count-of-most-popular-words-in-a-pandas-dataframe?rq=1


# In[55]:


policy = pd.read_csv('policy.csv')
policy.info()
policy.head()


# In[56]:


policy.fillna('none',inplace=True)
policy.head()


# In[57]:


ny = policy[policy['Location'] == 'New York']
ny.head()


# In[59]:


top_N = 10

stopwords = nltk.corpus.stopwords.words('english')

print(' '.join(stopwords))


# In[60]:


# RegEx for stopwords
RE_stopwords = r'\b(?:{},;)\b'.format('|'.join(stopwords))

# replace '|'-->' ' and drop all stopwords
words = (ny.Comments
           .str.lower()
           .replace([r'\|', RE_stopwords], [' ', ''], regex=True)
           .str.cat(sep=' ')
           .split()
)


# In[61]:


# generate DF out of Counter
rslt = pd.DataFrame(Counter(words).most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')

print('all frequencies, not including stopwords: ')

print('=' * 60)
print(rslt)
print('=' * 60)


# In[63]:


words = (ny[ny['Comments']!='none'].Comments
           .str.lower()
           .replace([r'\|', RE_stopwords], [' ', ''], regex=True)
           .str.cat(sep=' ')
           .split()
)

rslt = pd.DataFrame(Counter(words).most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')

print('filtered frequencies, not including stopwords: ')

print('=' * 60)
print(rslt)
print('=' * 60)


# In[64]:


# plot
rslt.plot.bar(rot=15, figsize=(16,10), width=0.8)


# In[ ]:


get_ipython().system("jupyter nbconvert --output-dir='output/' --to pdf new-york-policy.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to markdown new-york-policy.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to html new-york-policy.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to python new-york-policy.ipynb")


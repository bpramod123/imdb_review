#!/usr/bin/env python
# coding: utf-8

# In[4]:


conda install matplotlib


# In[14]:


import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
df=pd.read_csv(r'C:\Users\gunda\Desktop\projs\movies.csv')


# In[8]:


df


# In[41]:


#dropping null value rows
df1 = df.dropna()
df1


# In[42]:


df1['budget']=df1['budget'].astype('int64')
df1['gross']=df1['gross'].astype('int64')


# In[43]:


df1


# In[44]:


#getting the top budget films 
df.sort_values(by=['budget'],inplace=False, ascending=False)


# In[49]:


#dropping duplicates

df1.drop_duplicates(inplace=True)


# In[55]:


#scatter plot of each movie's budget vs gross

plt.scatter(x=df1['budget'], y=df1['gross'])

plt.title('budget vs gross')
plt.ylabel('budget')
plt.xlabel('gross')
plt.show()


# In[52]:


df.head()


# In[53]:


df.tail()


# In[63]:


#using seaborn bud vs gross
sns.regplot(x='budget',y='gross', data=df1)


# In[64]:


#correlation
df1.corr()


# In[65]:


df1.setindex("score")
df1


# In[67]:


df1.set_index("budget", inplace = True)


# In[68]:


df1


# In[69]:


df.corr(method='spearman')


# In[70]:


corr_mat=df1.corr(method='pearson')
sns.heatmap(corr_mat, annot=True)
plt.show()


# In[76]:


df1_numerized=df1

for col_name in df1_numerized.columns:
    if(df1_numerized[col_name].dtype=='object'):
        df1_numerized[col_name]=df1_numerized[col_name].astype('category')
        df1_numerized[col_name]=df1_numerized[col_name].cat.codes
df1_numerized


# In[78]:


corr_mat=df1_numerized.corr(method='pearson')
sns.heatmap(corr_mat, annot=True)
plt.title("Correlation matrix")
plt.xlabel("Movie features")
plt.xlabel("Movie features")
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Importing primary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import  warnings
warnings.filterwarnings('ignore')


# ## """(1) Data loading"""

# In[3]:


df  = pd.read_csv('C:\\Users\\Akshay\\Downloads\heart.csv',encoding = 'latin-1')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().values.any()


# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().any()


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.shape


# In[13]:


df.describe()


# ## How Many People Have Heart Disease, And How Many Don't Have Heart Disease In This Dataset? 

# In[16]:


df.columns


# In[15]:


plt.figure(figsize=[17,6])
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[18]:


df['target'].value_counts()


# In[19]:


plt.figure(figsize=[3,4])
sns.countplot(x='target',data=df)
plt.title('No of people have heart disease')
plt.show()


# ## Find Count of Male & Female in this Dataset 

# In[21]:


df['sex'].value_counts()


# In[22]:


df.groupby(['sex','target']).size()


# In[23]:


plt.figure(figsize=[5,4])
sns.countplot(x='sex',hue='target',data=df)
plt.xticks((0,1),('female','male'))
plt.legend(labels=['no_disease','disease'])
plt.xlabel('Gender',fontsize=12)
plt.ylabel('No of count',fontsize=12)
plt.title('Gender vs. disease/no_disease',fontsize=15)
plt.show()


# ## Check Age Distribution In The Dataset

# In[25]:


sns.distplot(df['age'])
plt.show()


# ## Check chest pain type 

# In[26]:


df['cp'].value_counts()


# In[27]:


plt.figure(figsize=[8,4])
sns.countplot(x='cp',data=df)
plt.xticks([0,1,2,3],['typical angina','atypical angina','non-anginal pain','asymptomatic pain'])
plt.xlabel('Chest pain',fontsize=12)
plt.ylabel('No of cp',fontsize=12)
plt.title('No and types of chest pains',fontsize=18)
plt.show()


# ## Show The Chest Pain Distribution As Per Target Variable

# In[29]:


df


# In[30]:


df.groupby(['target','cp']).size()


# In[31]:


sns.countplot(x='fbs',hue='target',data=df)
plt.legend(labels=['no_disease','disease'])
plt.show()


# ##   Check Resting Blood Pressure Distribution

# In[32]:


sns.boxplot(df['trestbps'])
plt.show()


# In[33]:


df['trestbps'].hist()


# # Compare Resting Blood Pressure As Per Sex Column

# In[34]:


df.columns


# In[35]:


g=sns.FacetGrid(df,hue='sex',aspect=4)
g.map(sns.kdeplot,'trestbps',shade=True)
plt.legend(labels=['male','female'])


# #  Show Distribution of Serum cholesterol

# In[36]:


df['chol'].hist()
plt.show()


# #  Plot Continuous Variables

# In[38]:


df


# In[40]:


cate_val = []
cont_val = []

for columns in df.columns:
    if df[columns].nunique() <= 10:
        cate_val.append(columns)
    else:
        cont_val.append(columns)


# In[41]:


cate_val


# In[42]:


cont_val


# In[43]:


df.hist(cont_val,figsize=[15,6])
plt.tight_layout()
plt.show()


# In[ ]:





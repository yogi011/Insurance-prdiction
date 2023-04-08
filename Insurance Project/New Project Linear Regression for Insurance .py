#!/usr/bin/env python
# coding: utf-8

# In[178]:


import pandas as pd # importing important Library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[179]:


df=pd.read_csv('Insurance.csv')#Reading Data set
df


# In[180]:


df.info() #info give the information about type of data


# In[181]:


df.columns # columns give information about colum name


# In[182]:


df.describe() # describe give all statical information 


# In[183]:


df.shape #shape give the total no of rows and columns


# In[184]:


df.head() #head print first 5 elements


# In[185]:


df.isna().sum() # isna().sum()give the total null values of datasets


# In[186]:


df.children.unique()#give the unique value from children colum


# In[187]:


df.smoker.value_counts() #value counts gives the values from smoker


# In[188]:


df.groupby(df['smoker']).max('charges')# group by groube the two columsn and give max values from charges


# In[189]:


df.region.unique() # find unique values from region colum


# In[190]:


sns.scatterplot(x='age',y='charges',hue='sex',data=df) #hue and style connect the relation ship between x and y


# In[191]:


sns.relplot(x='age',y='charges',col='sex',style='smoker',data=df) #relplot make relationship between xandy


# In[192]:


sns.relplot(x='age',y='charges',hue='sex',style='smoker',data=df,palette='husl')#palette is a color style combination of RGB


# In[193]:


sns.relplot(x='age',y='charges',col='sex',style='region',data=df)


# In[194]:


plt.bar(df['smoker'],df['charges'])# plot bar plot between smoker and charges
plt.show()


# In[195]:


plt.bar(df['region'],df['charges']) #plot bar plot between region and charges
plt.show()


# In[196]:


df.head()


# In[197]:


plt.hist(df['charges'])# hist plot for charges
plt.show()


# In[198]:


sns.scatterplot(x='bmi',y='charges',hue='sex',data=df)
plt.show()


# In[199]:


sns.relplot(x='bmi',y='charges',col='sex',data=df)
plt.show()


# In[200]:


sns.relplot(x='bmi',y='charges',col='smoker',data=df)
plt.show()


# In[201]:


sns.relplot(x='bmi',y='charges',col='sex',data=df)
plt.show()


# In[202]:


sns.boxplot(df['charges'])
plt.show()


# In[203]:


sns.boxplot(df['age'])
plt.show()


# In[204]:


sns.boxplot(df['bmi'])
plt.show()


# # Data Preprocessing

# # some data points are categorical then we have to convert into numerical for make a model and processes data

# In[205]:


df1=pd.get_dummies(df['region'],prefix='region',drop_first=True)
df1


# In[206]:


df=pd.concat([df,df1],axis=1).drop(['region'],axis=1)
df


# In[207]:


df2=pd.get_dummies(df['smoker'],prefix='smoker',drop_first=True)
df=pd.concat([df,df2],axis=1).drop(['smoker'],axis=1)
df


# In[208]:


df3=pd.get_dummies(df['sex'],prefix='sex',drop_first=True)
df=pd.concat([df,df3],axis=1).drop(['sex'],axis=1)
df


# In[209]:


df.head()


# In[210]:


df.corr()


# In[211]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='cool')


# # Linear Regression model

# In[212]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[213]:


x=df.drop(['charges'],axis=1)
y=df['charges']


# In[214]:


x


# In[215]:


y


# In[216]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)


# In[217]:


x_train


# In[218]:


y_train


# In[219]:


x_test


# In[220]:


y_test


# In[221]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[222]:


y_pred=lr.predict(x_test)
y_pred


# In[223]:


print(lr.intercept_)
print(lr.coef_)


# In[224]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import math


# In[225]:


print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))


# In[226]:


y_predict=lr.predict(x_test)
test_score=r2_score(y_test,y_predict)
print(test_score)


# In[227]:


y_test


# In[ ]:





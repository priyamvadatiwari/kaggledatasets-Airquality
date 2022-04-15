#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


#Q1: Read the airquality.csv data set.

df = pd.read_csv("airquality.csv")
df.head(20)


# In[4]:


#Q1.a:How many missing values are present for each variable?

df.isnull().sum()


# In[5]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


#1b. Create a data frame of complete cases and find the mean temperature using listwise deletion
df2 = df.copy()

df2.dropna(axis = 0, how = 'any', inplace = True)

df2['Temp'].mean()


# In[5]:


#1c. Find the mean temperature using pairwise deletion

df2.groupby((df2['Temp'])).mean()


# In[6]:


#1d: Which rows contain missing temperature values?

df[df['Temp'].isnull()]


# In[7]:


#1e:Create a box plot for the air quality data
df.boxplot(figsize = (5,5))

'''There is a difference in the mean temperature because in the listwise deletion, 
it deletes the entire row which has the missing values for any column. 
This could affect the Temperature column and its mean'''


# In[32]:


#Check for any outliers (using the default 1.5 IQR setting)? 
#What are the ozone outlier values? Create a new data frame called ozone_complete 
#that has all rows with ozone outliers removed.

Q1=df['Ozone'].quantile(0.25)
Q3=df['Ozone'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1 - 1.5 * IQR
upper_bound=Q3 + 1.5 * IQR
print(lower_bound,upper_bound)
#Outlier values - Lower bound = -49.875 ; Upper bound = 131.125

ozone_complete = df[(df['Ozone'] > lower_bound) & (df['Ozone'] < upper_bound)]


# In[9]:


#Question 2: Using the original airqualty.csv


# In[10]:


#a.

air_median= pd.read_csv("airquality.csv")

plt.figure(figsize=(12, 7))
sns.boxplot(x='Month',y='Solar.R',data=air_median,palette='summer')


# In[11]:


median =air_median['Solar.R'].median()

def impute_R(cols):
    Solar = cols[0]
    
    if pd.isnull(Solar):
            return median

    else:
        return Solar
air_median['Solar.R'] = air_median[['Solar.R']].apply(impute_R,axis=1)

air_median['Solar.R'].head(10)


# In[12]:


#2b: Create a new data set called air_mean from the air_median data set. 
#Impute the missing temperature values with the mean temperature for the month that the temperature 
#is missing from in the air_mean data set. For example, impute missing month 5 temperature values 
#with the mean of the non-missing temperatures for month 5.

air_mean = air_median[['Temp', 'Month','Ozone','Wind']].copy()


# In[13]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Month',y='Temp',data=air_mean, palette='autumn')


# In[14]:


#Caluclating mean for every month

df2=air_mean[air_mean['Month']==5]['Temp']
mean1 = df2.mean()

df3=air_mean[air_mean['Month']==6]['Temp']
mean2 = df3.mean()

df4=air_mean[air_mean['Month']==7]['Temp']
mean3 = df4.mean()

df5=air_mean[air_mean['Month']==8]['Temp']
mean4 = df5.mean()

df6=air_mean[air_mean['Month']==9]['Temp']
mean5 = df6.mean()


# In[15]:


def impute_Temp(cols):
    Temp = cols[0]
    Month = cols[1]
    
    if pd.isnull(Temp):

        if Month == 5:
            return mean1

        elif Month == 6:
            return mean2
        elif Month == 7:
            return mean3
        elif Month == 8:
            return mean4
        
        else:
            return mean5

    else:
        return Temp

air_mean['Temp'] = air_mean[['Temp','Month']].apply(impute_Temp,axis=1)


air_mean.head(10)


# In[24]:


#2c:Create a new data set called air_ratio from the air_mean data set. 
#Impute the missing values of the Ozone variable using ratio imputation in the air_ratio data set 

air_ratio = air_mean[['Ozone', 'Month','Wind']].copy()

ozone_mean=air_ratio.groupby('Month')['Ozone'].mean()

unique=air_ratio.groupby('Month')['Ozone'].nunique()

ratio=unique/ozone_mean

ratio


# In[25]:


m1 = 0.889251
m2 = 0.305660
m3 = 0.405986
m4 = 0.400257
m5 = 0.667763


# In[26]:


def impute_Ozone(cols):
    Ozone = cols[0]
    Month = cols[1]
    
    if pd.isnull(Ozone):

        if Month == 5:
            return m1

        elif Month == 6:
            return m2
        elif Month == 7:
            return m3
        elif Month == 8:
            return m4
        
        else:
            return m5

    else:
        return Ozone

air_ratio['Ozone'] = air_ratio[['Ozone','Month']].apply(impute_Temp,axis=1)


air_ratio.head(30)


# In[27]:


#2d:Create a new data set called air_complete from the air_ratio data set. 
#Use linear regression to impute the missing values of Wind using Ozone as 
#the independent variable in the air_complete data set.

air_complete = air_ratio[[ 'Ozone','Wind']].copy()
air_complete


# In[28]:


def random_imputation(air_ratio, Wind):

    data_missing = air_complete[Wind].isnull().sum()
    observed_values = air_complete.loc[air_complete[Wind].notnull(), Wind]
    air_complete.loc[air_complete[Wind].isnull(), Wind + '_imp'] = np.random.choice(observed_values, data_missing, replace = True)
    
    return air_complete


# In[31]:


missing_columns = ['Wind']
for a in missing_columns:
    air_complete[a + '_imp'] = air_complete[a]
    air_complete = random_imputation(air_complete, a)

air_complete.head(20)


# In[ ]:





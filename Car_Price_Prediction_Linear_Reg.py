#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries and Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


    # Importing dataset from https://www.kaggle.com/yogidsba/predict-used-car-prices-linearregression/data


# In[3]:


df = pd.read_csv('used_cars_data.csv')


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


# removing duplicate index.


# In[8]:


df=df.iloc[:,1:]
df.head()


# In[9]:


df.describe()


# In[10]:


df.shape


# In[11]:


print(df['Location'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner_Type'].unique())


# 

# ## Feature Engineering

# In[12]:


# finding null values, droping them and reseting index.


# In[13]:


df.isnull().sum()


# In[14]:


print("Shape of df Before dropping any Row: ",df.shape)
df = df[df['Mileage'].notna()]
print("Shape of df After dropping Rows with NULL values in Mileage: ",df.shape)
df = df[df['Engine'].notna()]
print("Shape of df After dropping Rows with NULL values in Engine : ",df.shape)
df = df[df['Power'].notna()]
print("Shape of df After dropping Rows with NULL values in Power  : ",df.shape)
df = df[df['Seats'].notna()]
print("Shape of df After dropping Rows with NULL values in Seats  : ",df.shape)
df = df[df['Price'].notna()]
print("Shape of df After dropping Rows with NULL values in Seats  : ",df.shape)


# In[15]:


df = df.reset_index(drop=True)


# In[16]:


df.head()


# In[17]:


# getting company name from name column and converting 'Mileage', 'Engine' and 'Power' column into float datatype.


# In[18]:


for i in range(df.shape[0]):
    df.at[i, 'Company'] = df['Name'][i].split()[0]
    df.at[i, 'Mileage(kmpl)'] = df['Mileage'][i].split()[0]
    df.at[i, 'Engine(CC)'] = df['Engine'][i].split()[0]
    df.at[i, 'Power(bhp)'] = df['Power'][i].split()[0]


# In[19]:


df['Mileage(kmpl)'] = df['Mileage(kmpl)'].astype(float)
df['Engine(CC)'] = df['Engine(CC)'].astype(float)


# In[20]:


x = 'n'
count = 0
position = []
for i in range(df.shape[0]):
    if df['Power(bhp)'][i]=='null':
        x = 'Y'
        count = count + 1
        position.append(i)
print(x)
print(count)
print(position)


# In[21]:


df = df.drop(df.index[position])
df = df.reset_index(drop=True)


# In[22]:


df.shape


# In[23]:


df['Power(bhp)'] = df['Power(bhp)'].astype(float)


# In[24]:


df.head()


# In[25]:


# converting 'New_Price' column into float datatype


# In[26]:


for i in range(df.shape[0]):
    if pd.isnull(df.loc[i,'New_Price']) == False:
        df.at[i,'New_car_Price'] = df['New_Price'][i].split()[0]


# In[27]:


df['New_car_Price'] = df['New_car_Price'].astype(float)


# In[28]:


df.head()


# In[29]:


# deleting useless features


# In[30]:


df.drop(["Name"],axis=1,inplace=True)
df.drop(["Mileage"],axis=1,inplace=True)
df.drop(["Engine"],axis=1,inplace=True)
df.drop(["Power"],axis=1,inplace=True)
df.drop(["New_Price"],axis=1,inplace=True)


# In[31]:


df.head()


# ## Data Visualization

# In[32]:


df.info()


# In[33]:


# plotting histogram of our target variable i.e 'Price' so that we can get glance on distribution of Car Price.


# In[34]:


df['Price'].describe()


# In[35]:


fig=plt.figure(figsize=(10,7))
plt.hist(df['Price'],bins=100)
plt.title("Histogram of Car Price")
plt.xlabel("Price")
plt.savefig('Hist_Car_Price.png')


# In[36]:


# Plotting box plot of Fuel_type vs Price to get insight about which type would cost more than other.


# In[37]:


df['Fuel_Type'].describe()


# In[38]:


fig=plt.figure(figsize=(10,7))
df1=pd.concat([df['Price'], df['Fuel_Type']],axis=1)
df1.boxplot(by="Fuel_Type", column=['Price']).set_title("")
plt.ylabel('Price')
plt.savefig('boxplot_fueltype.png')


# In[39]:


# Plotting box plot of Owner_type vs Price to know relation between type of owner(i.e first owner, second owner, etc) and Car Price.


# In[40]:


df['Owner_Type'].describe()


# In[41]:


fig=plt.figure(figsize=(10,7))
df1=pd.concat([df['Price'], df['Owner_Type']],axis=1)
df1.boxplot(by="Owner_Type", column=['Price']).set_title("")
plt.ylabel('Price')
plt.savefig('boxplot_ownertype.png')


# In[42]:


# Plotting bar graph of company vs count of cars. For example Maruti is most comman brand followed by Hyundai.


# In[43]:


df['Company'].describe()


# In[44]:


fig=plt.figure(figsize=(10,7))
df['Company'].value_counts().plot(kind='bar')
plt.savefig('carcnts_by_comp.png')


# ## working with categorical data

# In[45]:


df.info()


# In[46]:


# As for now there are five categorical features.
#1.Location
#2.Fuel_Type
#3.Transmission
#4.Owner_Type
#5.Company


# In[47]:


# Dividing these each features into categories and generating new columns.


# In[48]:


df=pd.get_dummies(df, columns=['Location', 'Fuel_Type'], drop_first=False)


# In[49]:


df.shape


# In[50]:


df=pd.get_dummies(df, columns=['Transmission'], drop_first=True)


# In[51]:


df.shape


# In[52]:


df


# In[53]:


df.replace({"First":1,"Second":2,"Third": 3,"Fourth & Above":4},inplace=True)
df.head()


# ## Feature Selection

# In[54]:


# Selecting final features that will be used for model building and droping all other useless feature.


# In[55]:


df.drop(["Company"],axis=1,inplace=True)


# In[56]:


df.drop(['New_car_Price'],axis=1, inplace=True)


# In[57]:


df


# In[58]:


df.columns


# In[59]:


y=df['Price']


# In[60]:


y.shape


# In[61]:


df.drop(['Price'],axis=1, inplace=True)


# In[62]:


x=df


# In[63]:


x.shape


# In[64]:


x


# ## model building

# In[65]:


# Building model using sklearn library.


# In[66]:


# First splitting data to train and test into 80:20 ratio for the model. 


# In[67]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[68]:


# Applying Linear Regression algorithm using sklearn library.


# In[69]:


from sklearn.linear_model import LinearRegression
multi_model = LinearRegression()
multi_model.fit(x_train, y_train)


# In[70]:


# Predicting y_test by giving x_test as an input to the model


# In[71]:


y_pred=multi_model.predict(x_test)


# In[72]:


# Model accuracy for x_train,y_train and x_test,y_test respectively. 


# In[73]:


multi_model.score(x_train,y_train)


# In[74]:


multi_model.score(x_test,y_test)


# In[ ]:





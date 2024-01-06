#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se


# In[2]:


sp = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\CODSOFT\advertising.csv")
sp.head()


# In[3]:


sp.shape


# In[4]:


sp.describe()


# In[39]:


se.pairplot(sp, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='hist')
plt.show()


# In[16]:


sp['TV'].plot.hist(bins=10, color="turquoise", xlabel="TV")


# In[9]:


sp['Radio'].plot.hist(bins=10, color="pink", xlabel="Radio")


# In[10]:


sp['Newspaper'].plot.hist(bins=10,color="beige", xlabel="newspaper")


# In[42]:


se.heatmap(sp.corr(),annot = True,cmap='coolwarm')
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sp[['TV']], sp[['Sales']], test_size = 0.3,random_state=0)


# In[19]:


print(X_train)


# In[20]:


print(y_train)


# In[21]:


print(X_test)


# In[22]:


print(y_test)


# In[23]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[24]:


res= model.predict(X_test)
print(res)


# In[44]:


print("Accuracy of the model: ", model.score(X_test,y_test)*100)


# In[26]:


model.coef_


# In[27]:


model.intercept_


# In[28]:


0.05473199* 69.2 + 7.14382225


# In[40]:


plt.style.use('Solarize_Light2')
plt.grid()
plt.plot(res)


# In[55]:


y_pred = model.predict(X_test)


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[58]:


# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[59]:


# Visualization of Predicted vs. Actual Sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs. Predicted Sales")
plt.show()


# In[ ]:





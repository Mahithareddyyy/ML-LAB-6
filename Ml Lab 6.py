#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdata.xlsx")
df


# In[ ]:


#A1


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


feature1 = df['embed_3']
feature2 = df['embed_4']
plt.scatter(feature1, feature2)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Scatter Plot of Feature1 vs Feature2')
plt.show()


# In[ ]:


#A2


# In[7]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#One feature is selected as an independent variable and one feature is selected as a dependent variable from the dataset
independent_feature = np.array(df['embed_2']).reshape(-1, 1)
dependent_feature = np.array(df['embed_4'])
# Creating a Linear Regression model
model = LinearRegression()
model.fit(independent_feature, dependent_feature)
# Predicting the values
predicted_values = model.predict(independent_feature)
# Calculating the mean squared error
mse = mean_squared_error(dependent_feature, predicted_values)
#Printing the mean squared error
print(f"Mean Squared Error: {mse:.2f}")


# In[ ]:


#A3


# In[9]:


# Plotting the linear regression line
plt.scatter(independent_feature, dependent_feature, label='Data points')
plt.plot(independent_feature, predicted_values, color='black', linewidth=3, label='Linear Regression')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()


# In[ ]:





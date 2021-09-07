#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')
suv_data = pd.read_csv(r'C:\Users\LENOVO\Desktop\machineLearngin\suv\suv_data.csv')
print(suv_data.head(20))
suv_data.shape


# In[84]:


sns.countplot(x = "Purchased", data = suv_data)


# In[85]:


suv_data["Age"].plot.hist()


# In[86]:


suv_data.info()


# In[87]:


suv_data.isnull().sum()


# In[10]:


sns.boxplot(x='EstimatedSalary', y = 'Purchased', data=suv_data)


# In[88]:


suv_data.head()
#sex = pd.get_dummies(suv_data["Gender"], drop_first=True)
#sex.head()


# In[91]:


X = suv_data.iloc[:,[2,3]].values
y = suv_data["Purchased"]


# In[95]:


y


# In[99]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[105]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[106]:


from sklearn.metrics import classification_report


# In[107]:


classification_report(y_test, y_pred)


# In[69]:


from sklearn.metrics import confusion_matrix


# In[110]:


confusion_matrix(y_test, y_pred)


# In[71]:


from sklearn.metrics import accuracy_score


# In[111]:


accuracy_score(y_test, y_pred)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ##  Importing required libraries and load the dataset

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\Jagan\Downloads\fraudTest.csv\fraudTest.csv")
df.head()


# In[82]:


df.shape


# In[78]:


df.info()


# In[76]:


df.describe()


# ##  Data preprocessing

# In[37]:


#checking the missing values
df.isnull().sum()


# In[39]:


#drop unwanted columns
df = df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'trans_num', 'dob'])
df.head()


# In[41]:


from sklearn.preprocessing import LabelEncoder
# Identify categorical columns for Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical columns
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])
df.head()    


# ##  Data Visualization

# In[67]:


# Generate a correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()


# ## Dividing the data into X-variable and y-variable

# In[43]:


#seperating features and target columns
x = df.drop('is_fraud',axis=1)
y=df['is_fraud']


# ##   splitting the data  

# In[45]:


#splitting the data into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 42)


# ## Train Model

# In[150]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.transform(x_test)


# ### Logistic Regression

# In[96]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l2',C = 0.01)
lr.fit(x_train,y_train)


# #### model prediction and model evaluation

# In[98]:


y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)


# In[100]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred,zero_division = 0))


# In[102]:


conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ### Random Forest

# In[49]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,random_state = 42)
model.fit(x_train_scaled,y_train)


# #### model prediction and model evaluation

# y_train_pred = model.predict(x_train_scaled)
# y_test_pred = model.predict(x_test_scaled)

# In[91]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))


# In[93]:


conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ### Decision Tree

# In[133]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(ccp_alpha = 0.01,random_state=42)
decision_tree.fit(x_train, y_train)


# #### model prediction and model evaluation

# In[135]:


y_train_pred = model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)


# In[137]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))


# In[139]:


conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[ ]:





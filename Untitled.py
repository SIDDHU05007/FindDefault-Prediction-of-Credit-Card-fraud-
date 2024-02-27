#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib


# In[2]:


# Load the dataset
data = pd.read_csv('creditcard.csv')


# In[3]:


# Display the first few rows of the DataFrame to verify the data loading
print(data.head())


# In[4]:


# Explore dataset's structure, features, and statistics
print("Dataset Structure:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())


# In[5]:


# Split data into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[6]:


# Initialize models
log_reg = LogisticRegression()
random_forest = RandomForestClassifier()



# In[7]:


# Train models
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)


# In[8]:


# Predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)



# In[9]:


# Evaluation
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))

print("Random Forest:")
print(classification_report(y_test, y_pred_random_forest))


# In[ ]:





# In[10]:


# Evaluate Logistic Regression
print("Confusion Matrix - Logistic Regression:")
print(confusion_matrix(y_test, y_pred_log_reg))



# In[11]:


# Evaluate Random Forest
print("Confusion Matrix - Random Forest:")
print(confusion_matrix(y_test, y_pred_random_forest))


# In[14]:


from sklearn.model_selection import GridSearchCV

# Define hyperparameters grid
param_grid = {'n_estimators': [10, 20],
              'max_depth': [None, 5]}



# In[15]:


# Grid Search for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search_rf.fit(X_train, y_train)


# In[16]:


import joblib

# Save the model to disk
filename = 'credit_card_fraud_detection_model.sav'
joblib.dump(grid_search_rf.best_estimator_, filename)

# Load the model from disk
loaded_model = joblib.load(filename)

# Make predictions using the loaded model
predictions = loaded_model.predict(X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





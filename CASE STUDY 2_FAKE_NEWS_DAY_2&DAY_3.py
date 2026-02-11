#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("fake_news_dataset.csv")
print("Data Size:", df.shape)


# In[2]:


df.info()


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


X=df.drop('Is_Fake' , axis=1)
y=df['Is_Fake']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


y_test.shape
X_train.shape


# In[7]:


X_test.shape


# In[8]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print("Trainig DAta Sizer:", X_train.shape)
print("Test data size:", X_test.shape)


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

log_model=LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred_log=log_model.predict(X_test_scaled)


# In[10]:


acc_log=accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", acc_log)
cm=confusion_matrix(y_test,y_pred_log)
print("Confusion Matrix: \n", cm)


# In[11]:


TP=cm[1,1]
TN=cm[0,0]
FP=cm[0,1]
FN=cm[1,0]


# In[13]:


print(TP)
print(TN)
print(FP)
print(FN)


# In[14]:


from sklearn.metrics import precision_score, recall_score
TP = 71
TN = 103
FP = 11
FN = 15
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")


# In[15]:


float(TP/(TP+FN))


# In[16]:


float(TN/(TN+FP))


# In[17]:


float(TP/(TP+FP))


# In[18]:


float(TP/(TP+FN))


# In[19]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

tree_model=DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, criterion="gini", random_state=42)


# In[20]:


tree_model.fit(X_train,y_train)
y_pred_tree=tree_model.predict(X_test)
acc_tree=accuracy_score(y_test,y_pred_tree)
print("Decision Tree Accuracy:", acc_tree)


# In[21]:


plt.figure(figsize=(12,6))
plot_tree(tree_model, feature_names=X.columns,class_names=['Real','Fake'],filled=True)
plt.show()


# In[22]:


from sklearn.ensemble import RandomForestClassifier

rf_model=RandomForestClassifier( n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features="sqrt", random_state=42)

rf_model.fit(X_train,y_train)



# In[23]:


y_pred_rf=rf_model.predict(X_test)
acc_rf=accuracy_score(y_test,y_pred_rf)
print("Random Forest Accuracy:", acc_rf)

importances=pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nMost Important clues for detecting Fake News:")
print(importances.sort_values(ascending=False))


# In[25]:


from sklearn.neighbors import KNeighborsClassifier

knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)

y_pred_knn=knn_model.predict(X_test_scaled)
acc_knn=accuracy_score(y_test,y_pred_knn)
print("K-NN Accuracy:", acc_knn)


# In[27]:


from sklearn.svm import SVC

svm_model=SVC(kernel='linear',random_state=42)
svm_model.fit(X_train_scaled,y_train)

y_pred_svm=svm_model.predict(X_test_scaled)
acc_svm=accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", acc_svm)


# In[28]:


from sklearn.naive_bayes import GaussianNB

nb_model=GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred_nb=nb_model.predict(X_test_scaled)
acc_nb=accuracy_score(y_test,y_pred_nb)
print("Naive Bayes Accuracy:", acc_nb)


# In[30]:


results=pd.DataFrame({
    'Model':['Logistic Regression','Decision Tree', 'Random Forest','k-NN','SVM','Naive Bayes'], 'Accuracy':[acc_log,acc_tree,acc_rf,acc_knn,acc_svm,acc_nb]
})

print("--- Final Model Comparison---")

print(results.sort_values(by='Accuracy', ascending=False))

print("\nConclusion:")
print("The Model with the Highest accuracy should be deployed for Connect All's")


# In[32]:


df.columns


# In[33]:


df['Is_Fake'].value_counts()


# In[ ]:





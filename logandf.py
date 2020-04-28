#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv("college_predictor.csv")
#plt.scatter(X[:,0], X[:,1])


# In[122]:


print(data[0:5])


# In[123]:


data.rename(columns={'Chance of Admit ': 'Chance of Admit'}, inplace=True)
a=data.columns



# In[124]:


def hasAdmitted(data):
    if data > 0.6:
        return 1
    else:
        return 0
data['Admit'] = data['Chance of Admit'].apply(hasAdmitted)
data.head(20)


# In[125]:


X=data.iloc[:,0:4]
X
Y=data.iloc[:,5:]
Y


# In[126]:


from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train,X_test,y_train,y_test=train_test_split(data.drop(['Admit'], axis=1),data['Admit'],
                                               test_size=0.3,random_state=50) # 70% training and 30% test

X_train


# In[127]:


from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=5)
s=rf.fit(X_train, y_train)
print(s)



# In[128]:


rf.feature_importances_


# In[129]:


s=rf.score(X_test, y_test)
print("therefore the random forest regreesion has a accuracy of ",s)


# In[130]:


data


# In[131]:


#data=data.drop('Admit', axis=1)
#data=data.drop('Serial No.',axis=1).columns


# In[132]:


feat_importances = pd.Series(rf.feature_importances_, index=data.drop('Admit', axis=1).columns)
feat_importances.sort_values(ascending=False).plot(kind='barh')


# In[133]:


from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(data.drop(['Admit'], axis=1),data['Admit'],
                                               test_size=0.3,random_state=50) # 70% training and 30% test
X_test
lr = LogisticRegression()
lr.fit(X_train, y_train)
f=lr.score(X_test,y_test)
#data.predict(X_test)
print("The Accuracy of Logistic Regression is:",f)

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
v=model.score(X_test,y_test)
print("The Accuracy of SVM is:",v)





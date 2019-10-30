#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,auc,roc_auc_score,precision_score,recall_score


# In[2]:


#importing datasets
train_data = pd.read_csv("train_2v.csv")
test_data = pd.read_csv("test_2v.csv")


# In[3]:


train_data.shape


# In[4]:


test_data.shape


# In[5]:


test_data.head()


# In[6]:


train_data.describe()


# In[7]:


#cleaning


# In[8]:


train_data.isnull().sum()/len(train_data)*100


# In[9]:


test_data.isnull().sum()/len(test_data)*100


# In[10]:


joined_data = pd.concat([train_data,test_data])


# In[11]:


print ('Joined Data Shape: {}'.format(joined_data.shape))


# In[12]:


joined_data.isnull().sum()/len(joined_data)*100


# In[13]:


train_data["bmi"]=train_data["bmi"].fillna(train_data["bmi"].mean())


# In[14]:


train_data.head()


# In[15]:


#handling categorical data


# In[16]:


label = LabelEncoder()
train_data['gender'] = label.fit_transform(train_data['gender'])
train_data['ever_married'] = label.fit_transform(train_data['ever_married'])
train_data['work_type']= label.fit_transform(train_data['work_type'])
train_data['Residence_type']= label.fit_transform(train_data['Residence_type'])


# In[17]:


train_data_with_smoke = train_data[train_data['smoking_status'].notnull()]


# In[18]:


train_data_with_smoke.head()


# In[19]:


train_data_with_smoke['smoking_status']= label.fit_transform(train_data_with_smoke['smoking_status'])


# In[20]:


train_data_with_smoke.head()


# In[21]:


train_data_with_smoke.shape


# In[22]:


train_data_with_smoke.drop(columns='id',axis=1,inplace=True)


# In[23]:


train_data_with_smoke.head()


# In[49]:


train_data_with_smoke['smoking_status'].value_counts()


# In[25]:


train_data_with_smoke.corr('pearson')


# In[46]:


train_data_with_smoke['work_type'].value_counts()


# In[27]:


#handling imbalanced data


# In[28]:


ros = RandomOverSampler(random_state=0)
smote = SMOTE()


# In[29]:


X_resampled, y_resampled = ros.fit_resample(train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'], 
                                            train_data_with_smoke['stroke'])


# In[30]:


print ('ROS Input Data Shape for Smoke Data: {}'.format(X_resampled.shape))
print ('ROS Output Data Shape for Smoke Data: {}'.format(y_resampled.shape))


# In[31]:


#train test split of balanced data


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# In[33]:


#implementing the model


# In[34]:


log = LogisticRegression(penalty='l2', C=0.1)
log.fit(X_train,y_train)

pred = log.predict(X_test)
print(classification_report(y_test,pred))
print (accuracy_score(y_test,pred))
print (confusion_matrix(y_test,pred))

precision = precision_score(y_test,pred)
recall = recall_score(y_test,pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

y_pred_proba = log.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
impFeatures = pd.DataFrame(log.coef_[0] ,index=train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)


# In[35]:


model = LogisticRegression(penalty='l2', C=0.1)
model.fit(X_train,y_train)


# In[36]:


#predicting class labels for test set
predicted = model.predict(X_test)


# In[37]:


#generating class probabilities for test set
probs = model.predict_proba(X_test)


# In[38]:


# generate evaluation metrics
print("Accuracy Score = ",metrics.accuracy_score(y_test, predicted))
print("AUC Score = ",metrics.roc_auc_score(y_test, probs[:, 1]))


# In[39]:


scores = cross_val_score(LogisticRegression(), X_resampled, y_resampled, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())


# In[40]:


array = np.array([(1,65.0,0,0,1,2,1,103.12,31.4,2), (0,45.0,1,0,1,1,0,89.82,28.4,1)])


# In[41]:


array


# In[42]:


newpred = model.predict(array)


# In[43]:


newpred


# In[44]:


new_predictions = model.predict_proba(array)


# In[45]:


new_predictions


# In[52]:


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[53]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


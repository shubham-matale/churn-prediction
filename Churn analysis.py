#!/usr/bin/env python
# coding: utf-8

# ## Loading Libraraies
#

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns


# Loading Data

# In[2]:


churn=pd.read_excel("churn.xlsx")
churn.head()


# # Data Exploration

# In[3]:


churn.info()


# Check for missing values

# In[4]:


#check missing values

churn.columns[churn.isnull().any()]


# Out of 21 features , none of them have missing values

# In[5]:


print("There are {} numeric and {} categorical columns in churn data".format(churn.select_dtypes(include=[np.number]).shape[1],churn.select_dtypes(exclude=[np.number]).shape[1]))


# In[6]:


churn.select_dtypes(exclude=[np.number]).head()


# The columns "state" and "phone" are not usefull in perdicting the target variable

# ### Data Preprocessing

# In[7]:


X=churn.iloc[:,4:20].values
Y=churn.iloc[:, 20].values


# converting catagorical data to numeric (using Label Encoding)

# In[8]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)


# Visualizing the correlation of data

# In[9]:


cor=churn.corr()
# sns.heatmap(cor)


# Scaling data by Standerd Scaler

# In[10]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# As you can see the DataConversionWarning it means our Data is know Scaled. lets visualize it.

# In[11]:





# ### Training model

# In[12]:


#importing labrararies for Artifical Nural network
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import load_model


# Making objects of our Classifier

# In[14]:

#
# clf=Sequential()
#
#
# # Building Layers for our ANN
#
# # In[15]:
#
#
# #adding layers to ANN
# clf.add(Dense(units=24,activation="relu",kernel_initializer="uniform",kernel_regularizer=regularizers.l2(0.001),input_dim=16))
# #adding two more hidden layer to ANN
# clf.add(Dense(units=24,activation="relu",kernel_initializer="uniform",kernel_regularizer=regularizers.l2(0.001)))
# clf.add(Dense(units=24,activation="relu",kernel_initializer="uniform",kernel_regularizer=regularizers.l2(0.001)))
# #adding output layer
# clf.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
# #compiling ANN
# clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
#
# # Fitting ANN
#
# # In[16]:
#
#
# history=clf.fit(X,Y,batch_size=20,epochs=250)
# clf.save('my_model.h5')
# clf.save('model.h5')
classifier=load_model('my_model.h5')
prediction = classifier.predict(X_test)
# Displaying curves of loss and accuracy during training

# In[17]:

# print(history.history['accuracy'])
# prediction= clf.predict(X_test)
print(prediction)

# testScore = 0
# trainScore = 0
# for i in range(0,len(prediction)):
#     if prediction[i]>0.5:
#         prediction[i] = 1
#     else:
#         prediction[i] = 0
#     print(prediction[i],y_test[i])
#     if prediction[i] != y_test[i]:
#         testScore+=1
#
# print(((len(y_test)-testScore)/len(y_test))*100)
# # train_acc = clf.evaluate(X_train, y_train, verbose=0)
# # test_acc = clf.evaluate(X_test, y_test, verbose=0)
# # print(train_acc, test_acc)
# # from sklearn.metrics import precision_score
# # precision_score(y_test,prediction)
# # acc = history.history['accuracy']
# # loss = history.history['loss']
# # epochs = range(1, len(acc) + 1)
# # plt.plot(epochs, acc, 'bo', label='Training acc')
# # plt.title('Training accuracy')
# # plt.legend()
# # plt.figure()
# # plt.plot(epochs, loss, 'bo', label='Training loss')
# # plt.title('Training loss')
# # plt.legend()
# # plt.show()
# #
# #
# # # we can conclude that the model got accuracy of about 97% and was still increasing at slow rate after 250 epochs
# #
# # # In[ ]:
# #
# #
# #
# #
# #
# # # In[ ]:
# #
# #
# #
# #

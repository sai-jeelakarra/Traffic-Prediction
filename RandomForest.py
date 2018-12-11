# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:40:12 2018

@author: saije
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training dataset
train_dataset = pd.read_csv('train_set.csv')
X_train = train_dataset.iloc[:, [2,3,4,5]].values
y_train = train_dataset.iloc[:, 6].values


# Importing the testing dataset
test_dataset = pd.read_csv('test_set.csv')
X_test = test_dataset.iloc[:, [2,3,4,5]].values
y_test = test_dataset.iloc[:, 6].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X_train = onehotencoder.fit_transform(X_train).toarray()

X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X_train = onehotencoder.fit_transform(X_train).toarray()

X_train[:, 4] = labelencoder_X.fit_transform(X_train[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X_train = onehotencoder.fit_transform(X_train).toarray()

X_train[:, 5] = labelencoder_X.fit_transform(X_train[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X_train = onehotencoder.fit_transform(X_train).toarray()


# Encoding categorical data
# Encoding the Independent Variable
labelencoder_XY = LabelEncoder()
X_test[:, 2] = labelencoder_XY.fit_transform(X_test[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X_test = onehotencoder.fit_transform(X_test).toarray()

X_test[:, 3] = labelencoder_XY.fit_transform(X_test[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X_test = onehotencoder.fit_transform(X_test).toarray()

X_test[:, 4] = labelencoder_XY.fit_transform(X_test[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X_test = onehotencoder.fit_transform(X_test).toarray()

X_test[:, 5] = labelencoder_XY.fit_transform(X_test[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X_test = onehotencoder.fit_transform(X_test).toarray()




# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rand_classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
rand_classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred1 = rand_classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_test, y_pred)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)


from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='micro') 













# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training dataset
dataset = pd.read_csv('train_set.csv')
X_train = dataset.iloc[:, [2,3,4,5]].values
y_train = dataset.iloc[:, 6].values


# Importing the testing dataset
dataset = pd.read_csv('test_set.csv')
X_test = dataset.iloc[:, [2,3,4,5]].values
#y_test = dataset.iloc[:, 6].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)


"""
# Making the Confusion Matrix
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""

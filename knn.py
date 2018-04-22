import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('heart.csv')
print (df)
X = df.iloc[:, :(-1)].replace('Present', '0').replace('Absent', '1')
print (X)

y = df.iloc[:, (-1)].values
print (y)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)

print (X_train.shape[0], X_test.shape[0])

tree = KNeighborsClassifier(n_neighbors = 2)
tree.fit(X_train, Y_train)

print (tree)
y_pre = tree.predict(X_test)
print ("Misclassifier samples :",sum((Y_test!=y_pre)))
print ("Accuracy of tree :",round(sum(Y_test==y_pre))/Y_test.shape[0],3)


print("Accuracy of tree :",accuracy_score(Y_test,y_pre))


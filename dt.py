import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import export_graphviz


df = pd.read_csv('heart.csv')
print (df)
X = df.iloc[:, :(-1)].replace('Present', '0').replace('Absent', '1')

print (X)
# X = min_max_scaler.fit_tranform(x)
y = df.iloc[:, (-1)].values
print (y)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
print("Number of training sample :",X_train.shape[0])
print("Number of testing sample :",X_test.shape[0])
print("Class lable :",np.unique(y))



tree = DecisionTreeClassifier(criterion ="entropy",max_depth = 3)
tree.fit(X_train,Y_train)

y_pre = tree.predict(X_test)
print ("Misclassifier samples :",sum((Y_test!=y_pre)))
print ("Accuracy of tree :",round(sum(Y_test==y_pre))/Y_test.shape[0],3)






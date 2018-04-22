import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D



df = pd.read_csv('heart.csv')
X = np.array(df[['alcohol', 'age']])
# X = min_max_scaler.fit_tranform(x)
y = np.array(df['chd'])

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33)

clf = SVC(kernel = 'rbf', C = 0.025)
clf.fit(X_train, Y_train)
# plt.scatter(X, marker = 'o', c = 'b')

#w = clf.coef_
b = clf.intercept_
ac = clf.score(X_test, Y_test)
print "Accuracy: %.2f %%" %(100*ac)

X1 = X[np.where(y==1)]
X0 = X[np.where(y==0)]
fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
plt.scatter(X1[:,0],X1[:,1], marker='^', c='b')
plt.scatter(X0[:,0],X0[:,1], marker='o', c='y')
plt.xlabel('alcohol')
plt.ylabel('age')
plt.legend(1,2)
plt.show()

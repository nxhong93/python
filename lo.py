from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


#dataset
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv('data2.csv')
X = np.array(df[['grade1','grade2']])
X = scaler.fit_transform(X)
Y = np.array(df['label;;;;'].map(lambda x: float(x.rstrip(';'))))


#train model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
mod = LogisticRegression()
mod.fit(X_train, Y_train)
result = mod.score(X_test,Y_test)
print("Accuracy: %.3f%%") % (result*100.0)

#show
plt.scatter(X[np.where(Y==1),0],X[np.where(Y==1),1], marker='o', c='b')
plt.scatter(X[np.where(Y==0),0],X[np.where(Y==0),1], marker='x', c='r')
plt.xlabel('grade1')
plt.ylabel('grade2')
plt.legend(['Not Admitted', 'Admitted'])
plt.show()
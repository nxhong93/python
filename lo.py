from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


#dataset
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv('php0iVrYT.csv')
X = np.array(df.iloc[:, :-1])
X = scaler.fit_transform(X)
Y = np.array(df.iloc[:,-1])



#train model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25, random_state = 1)
mod = LogisticRegression()
mod.fit(X_train, Y_train)
result = mod.score(X_test,Y_test)
print(("Accuracy: %.3f%%") % (result*100.0))
print(mod.predict([[22,1,20,70]]))

#show
# plt.scatter(X[np.where(Y==2),0],X[np.where(Y==2),1], marker='o', c='b')
# plt.scatter(X[np.where(Y==1),0],X[np.where(Y==1),1], marker='x', c='r')
# plt.legend(['yes', 'No'])
# plt.show()


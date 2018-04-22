from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



#dataset
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv('php0iVrYT.csv')
print (df.describe())
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()

df.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
plt.show()

X0 = np.array(df.iloc[:, :-1])
X = scaler.fit_transform(X0)
y = np.array(df.iloc[:,-1])

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)


mod = []
mod.append(('LR', LogisticRegression()))
mod.append(('LDA', LinearDiscriminantAnalysis()))
mod.append(('KNN', KNeighborsClassifier()))
mod.append(('CARD', DecisionTreeClassifier()))
mod.append(('NB', GaussianNB()))
mod.append(('SVM', SVC()))


re = []
na = []
for name, model in mod:
	k = model.fit(X_train, Y_train)
	cv_re = k.score(X_test, Y_test)
	print((name + ": %.3f%%") % (cv_re*100.0))
	re.append(cv_re)
	na.append(name)

print (("KQ: %.3f%% (%f)") % (100*np.mean(re), np.std(re)))
print (("Tot nhat: %.3f%%") % (100*np.max(re)))

#show
# plt.scatter(X[np.where(Y==2),0],X[np.where(Y==2),1], marker='o', c='b')
# plt.scatter(X[np.where(Y==1),0],X[np.where(Y==1),1], marker='x', c='r')
# plt.legend(['yes', 'No'])
# plt.show()


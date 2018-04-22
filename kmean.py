from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Importing the dataset
dt = pd.read_csv('xclara.csv')
X = np.mat(dt)



plt.plot(X[:,0] ,X[:,1] ,'ro')

def dist(a, b, ax = 1):
    return np.linalg.norm(a - b, axis = ax)

k = 3
x1 = np.mat(np.random.randint(0, np.max(X) - 20, size=k))
print x1
y1 = np.mat(np.random.randint(0, np.max(X) - 20, size=k))

x_ = np.array(np.hstack((x1.T, y1.T)))

plt.plot(x1,y1,'^b')

x_old = np.zeros(x_.shape)
cl = np.zeros(len(X))

er = dist(x_, x_old, None)

while er != 0:
    for i in range(len(X)):
        dis = dist(X[i], x_)
        clu = np.argmin(dis)
        cl[i] = clu

    x_old = deepcopy(x_)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if cl[j] == i]
        x_[i] = np.mean(points, axis=0)
        print x_[i]

    er = dist(x_, x_old, None)
print er  

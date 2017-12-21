from sklearn import linear_model
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


#import csv
df = pd.read_csv('Salary_Data.csv')
x = np.matrix(df['YearsExperience']).T
y = np.matrix(df['Salary']).T
#df = np.loadtxt('Salary_Data.csv', delimiter=',', skiprows=1,unpack=False)
#x = df[:,:-1]
#y = df[:,1:]
#print x
#print y

# Calculating w
one = np.ones((x.shape[0], 1))
Xbar = np.hstack((one, x))
A = Xbar.T*Xbar
b = Xbar.T*y
w = A.I*b
print w

# Preparing the fitting line 
w0 = w[0][0]
w1 = w[1][0]
x0 = np.linspace(0, 12,2)
y0 = w0 + w1*x0

# Drawing the fitting line 
plt.plot(x.T, y.T, 'ro')
plt.plot(x0, y0.T, 'b')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
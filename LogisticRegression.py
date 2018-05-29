import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import math

def mean_normalization(X):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    X_normalization = (X - mean) / stdev

    return X_normalization

df = pd.read_csv("label_data.csv", header=0)
df.columns = ["grade1","grade2","label"]


X = df[["grade1","grade2"]]
X = np.array(X)
X = mean_normalization(X)
bias_feature = np.zeros(len(X)).reshape(len(X), 1)
bias_feature[:] = 1
X = np.hstack((bias_feature,X))

Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)

print(X)
print(Y)



def sigmoid(x):
    return 1/(1+np.exp(-x))


def hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i] * theta[i]
    return sigmoid(z)


def cost_Function(X,Y,theta):
    m = len(X)
    sumOfErrors = 0
    for i in range(m):
        xi = X[i]
        h = hypothesis(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(h)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-h)
    sumOfErrors += error
    J = -1/m * sumOfErrors
    return J


def Cost_Function_Derivative(X,Y,theta,j,alpha):
    sumErrors = 0
    m = len(X)
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hi = hypothesis(theta,X[i])
        error = (hi - Y[i])*xij
        sumErrors += error
    J = (alpha/m) * sumErrors
    return J

def Gradient_Descent(X,Y,theta,alpha):
    new_theta = []
    for j in range(len(theta)):
        CFDerivative = Cost_Function_Derivative(X,Y,theta,j,alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta

costs = []
def Logistic_Regression(X,Y,alpha,theta,num_iters):
    for x in range(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,alpha)
        theta = new_theta
        if x % 20 == 0:
            cost = cost_Function(X,Y,theta)
            print('step ', x)
            print('theta ', theta)
            print('cost ', cost)
            costs.append(cost)
            print()
    return costs

initial_theta = [0,0,0]
alpha = 0.01
iterations = 1000

costs2 = Logistic_Regression(X,Y,alpha,initial_theta,iterations)

x_axis = []
for i in range(0,1000,20):
    x_axis.append(i)

plt.plot(x_axis, costs2, 'r')
plt.xlabel('Number of Epochs')
plt.ylabel('cost')
plt.show()

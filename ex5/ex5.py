import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import seaborn as sns 

## Import the data
data = loadmat('ex5data1.mat')
X_train, y_train = data['X'], data['y'] # 12x1 , 12x1
X_val, y_val = data['Xval'], data['yval'] # 21x1 , 21x1
X_test, y_test = data['Xtest'], data['ytest'] # 21x1 , 21x1

# Insert column of ones in the first column of X_train
X = np.insert(X_train, 0, 1, axis=1) # 12x2

## Visualize training data
plt.scatter(X_train, y_train, c='r', marker='x')
# plt.show()

def linearRegCostFunction(theta, X, y_train, regulizer_lambda):
	m = X.shape[0] # 12 - scalar
	h = np.dot(X, theta) # 12x2 2x1 = 12x1

	J = (1 / (2 * m)) * (np.sum(np.square(h - y_train))) + (regulizer_lambda / (2 * m)) * (np.sum(np.square(theta[1:])))
	
	return(J)
	
def gradient(theta, X, y_train, regulizer_lambda):
	m = X.shape[0] # 12 - scalar
	h = np.dot(X, theta.reshape(-1, 1)) # 12x2 2x1 = 12x1

	grad = (1 / m)* (np.dot(X.T, (h - y_train))) + ((regulizer_lambda / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)])

	return(grad.flatten())


theta = np.ones((X.shape[1],1))
regularizer = 1
J = linearRegCostFunction(theta, X, y_train, regularizer)
grad = gradient(theta, X, y_train, regularizer)

print(J, grad)


## Train the linear regression: minimize the cost function to find optimized theta
import scipy.optimize 
def train(X, y_train, regulizer_lambda):
	initial_theta = np.array([[11], [10]])
	# scipy minimization flattens the x0 initial guess all the way through. So it required both instances of reshaping of theta in the gradient function
	optimization = scipy.optimize.minimize(linearRegCostFunction, initial_theta, args=(X, y_train, regularizer), method=None, jac=gradient)

	return(optimization)

theta1, theta2 = train(X, y_train, 0).x

from sklearn.linear_model import LinearRegression
lr_x = np.arange(np.amin(X_train), np.amax(X_train))
lr_y = theta1 + (lr_x * theta2)
plt.scatter(X_train, y_train, c='r', marker='x')
plt.plot(lr_x, lr_y)
plt.show()

def learningCurve(X, y_train, X_val, y_val, reg):
	m = y_train.size
	training_err = np.ones((m,1))
	cv_err = np.ones((m,1))
	for i in np.arange(2, m):
		theta = train(X[:i+1], y_train[:i+1], reg)
		training_err[i] = linearRegCostFunction(theta.x, X[:i], y_train[:i], reg)
		cv_err[i] = linearRegCostFunction(theta.x, X_val, y_val, reg)
	return(training_err, cv_err)

# training_err, cv_err = learningCurve(X, y_train, X_val, y_val, 0)

# plt.plot(np.arange(12), training_err)
# plt.plot(np.arange(12), cv_err)
# plt.show()

''' 
ERROR that it's throwing in line 22
shapes (21,1) and (2,) not aligned. 
I tried T and double transposing, and reshaping. It doesn't take any of those manipulations.
It must be a reshaping issue though.
'''

# def polyFeatures(X, degree):

# 	for i in np.arange(2, degree+1):
# 		X = np.c_[[X],[X[:,0]] ** i]
# 	return(X)

# polyFeatures(X_train, 3)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

poly = PolynomialFeatures(degree=8)
X_train_poly = poly.fit_transform(X_train.reshape(-1,1))

lin_regr = LinearRegression()
lin_regr.fit(X_train_poly, y_train)

xaxis = np.linspace(np.amin(X_train), np.amax(X_train))

ploty = lin_regr.intercept_ + np.sum(lin_regr.coef_ * poly.fit_transform(xaxis.reshape(-1, 1)), axis=1)
plt.plot (xaxis, ploty)
plt.scatter(X_train, y_train)
plt.show()


# web driver: https://www.youtube.com/watch?v=bhYulVzYRng

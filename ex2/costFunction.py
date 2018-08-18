import numpy as np
import sigmoid

def costFunction(X, y, theta):
	h = sigmoid.sigmoid(np.dot(X, theta))
	m = len(y)
	J = 0
	
	J = (1./m) * ((np.dot((-y).T, np.log(h))) - np.dot((1. - y).T, np.log(1. - h)))
	
	return J

def gradient(X, y, theta):
	m = len(y)
	grad = np.zeros((2,1))
	grad = (1./m) * np.dot(X.T, ((sigmoid.sigmoid(np.dot(X, theta))) - y))
	grad = grad.flatten()

	return grad


def costFunctionReg(X, y, theta, lam):
	h = sigmoid.sigmoid(np.dot(X, theta))
	m = len(y)
	J = 0
	
	J = ((1./m) * ((np.dot((-y).T, np.log(h))) - np.dot((1. - y).T, np.log(1. - h)))) + (lam/(2.*m))*np.sum(np.square(theta[1:]))
	
	return J


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat

data = loadmat('ex4data1.mat')
X = np.c_[np.ones((data['X'].shape[0], 1)), data['X']]
y = data['y']
weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
parameters = np.r_[theta1.ravel(), theta2.ravel()]

# Looking at 20 images from the data set. Start by initializing a matrix of 20 random images from the data
X_im = np.zeros((20, 401))
for i in range(20):
	row = np.random.randint(0, X.shape[0])
	X_im[i] = X[row]

def plot_figures(iterations, nrows, ncols):
	fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', sharey='all')
	for ind in range(iterations):
		axeslist.ravel()[ind].imshow(np.reshape(X_im[ind, 1:], (20, 20)).T, cmap='gray')
	return(fig, axeslist)
number_of_im = 20
plot_figures(number_of_im, 5, 4)
plt.show()

'''
The three most common activation functions in NNs are the identity function, the logistic sigmoid function,
and the hyperbolic tangent function. We're using teh logistic sigmoid function here. The gradient of this function
is just the derivative, and is necessary for backpropagation through the NN.
'''
def sigmoid(z):
	return(1 / (1 + np.exp(-z)))

def sigmoidGradient(z):
	g = sigmoid(z) * (1 - sigmoid(z))

	return(g)

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg):
	#reshape nn_params (which are just theta1 and theta2)
	m = X.shape[0]
	theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))

	theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1))

	# Turning y into Y
	Y = pd.get_dummies(y.ravel()).values

	# Part 1: Forward propagation
	a1 = X
	z2 = np.dot(theta1, a1.T) 
	a2 = np.c_[np.ones((X.shape[0], 1)), sigmoid(z2.T)] 
	z3 = np.dot(theta2, a2.T) 
	a3 = sigmoid(z3) # should be 10x5000?

	# Calculate cost function J
	J = -1 * (1 / m) * np.sum((np.log(a3.T) * Y + np.log(1 - a3).T * (1 - Y))) + (reg / (2 * m)) * (np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))

	# Discover all 3 sigmas
	sigma3 = a3.T - Y 
	sigma2 = np.dot(theta2[:, 1:].T, sigma3.T) * sigmoidGradient(z2)

	# Calculate the capital deltas
	delta_1 = np.dot(sigma2, a1) 
	delta_2 = np.dot(sigma3.T, a2) 

	# Regularized gradient
	p1 = np.c_[np.ones((theta1.shape[0], 1)), theta1[:, 1:]]
	p2 = np.c_[np.ones((theta2.shape[0], 1)), theta2[:, 1:]]

	Theta1_grad = delta_1 / m + (p1 * reg) / m
	Theta2_grad = delta_2 / m + (p2 * reg) / m

	return(J, Theta1_grad, Theta2_grad)


J, Theta1_grad, Theta2_grad = nnCostFunction(parameters, 400, 25, 10, X, y, 0)

print(J, Theta1_grad, Theta2_grad)
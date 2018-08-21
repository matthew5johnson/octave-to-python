import numpy as np 
import matplotlib.pyplot as plt
import scipy.io 



### Loading the data from matlab formatted files. It's already tagged, and scipy puts the data into dictionary format
data = scipy.io.loadmat('ex3data1.mat')
weights = scipy.io.loadmat('ex3weights.mat')
y = data['y']
X = np.c_[(np.ones((data['X'].shape[0], 1))), data['X']]


### Reproducing images from the data
# Let's make the first row of X (400 pixels) into a grayscale picture that's 20 x 20.
# 1. pick a random row from the dat and reshape it into a 20x20 matrix; 2. plot the matrix with imshow from matplotlib; 3. assign grayscale spectrum to values
row_select = np.random.randint(1, X.shape[0])
X_work = X[row_select, 1:]
test = np.reshape(X_work, (20,20), 'F')
plt.imshow(test, cmap='Greys')
#plt.show()



def sigmoid(z):
	return(1 / (1 + np.exp(-z)))

# Needs to be well vectorized so that it can avoid any for loops 
def lrCostFunction(theta, reg, X, y):
    m = y.size
    h = sigmoid(np.dot(X, theta))
    
    J = -1 * (1/m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg / (2 * m)) * np.sum(np.square(theta[1:]))
    
    return(J)

def lrGradient(theta, reg, X, y):
    m = y.size
    h = sigmoid(np.dot(X, theta.reshape(-1, 1)))  # reshape theta into a vector
    
    grad = (1 / m) * X.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]  # the last .reshape makes theta into a vector
    
    return(grad)


# Import scipy's optimization library to use the minimization package within our oneVsAll function
from scipy.optimize import minimize

'''
We only explictly call oneVsAll and predictOneVsAll. The sigmoid, Cost, and Gradient functions above are all used in these 2 functions below
'''

def oneVsAll(features, classes, num_labels, reg):
    initial_theta = np.zeros((X.shape[1], 1))  
    all_theta = np.zeros((num_labels, X.shape[1]))

    # For each class, 1-10, we find the parameters/thetas by minimizing over theta within the cost function, using the gradient function as the jacobian
    for clss in np.arange(1, num_labels + 1):
      minim = minimize(lrCostFunction, initial_theta, args=(reg, features, (classes == clss) * 1), method='SLSQP', jac=lrGradient)
      all_theta[clss - 1] = minim.x
    return(all_theta)

# We need the oneVsAll function in order to find the theta parameter used by the predictOneVsAll function
theta = oneVsAll(X, y, 10, 0.1)

def predictOneVsAll(all_theta, features):
    probs = sigmoid(np.dot(X, all_theta.T))
    
    # adding one to sync up classes with python indexing
    return(np.argmax(probs, axis=1) + 1)
   
pred = predictOneVsAll(theta, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel()) * 100))
   

##### Now onto the NN forward propagation part of the problem. Huge thanks to JWarmenhoven's example on github for helping me through this part. Great resource
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C=10, penalty='l2', solver='liblinear')
# Scikit-learn fits intercept automatically, so we exclude first column with 'ones' from X when fitting.
classifier.fit(X[:, 1:], y.ravel())


### Grab thetas from the data
theta1, theta2 = weights['Theta1'], weights['Theta2']


pred2 = classifier.predict(X[:, 1:])
print('Training set accuracy: {} %'.format(np.mean(pred2 == y.ravel()) * 100))

def predict(theta_1, theta_2, features):
    z2 = theta_1.dot(features.T)
    a2 = np.c_[np.ones((data['X'].shape[0], 1)), sigmoid(z2).T]
    
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)
    
    return(np.argmax(a3, axis=1) + 1) 
   
pred = predict(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel()) * 100))
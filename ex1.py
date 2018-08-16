import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

### Part 2: Plotting
filename = "ex1data1.txt"
raw_data = np.loadtxt(filename, delimiter=',')   #how to load the data with pd: pd.read_csv(filename, header=None, usecols=[0,1])


### Part 3: Cost and Gradient Descent
X = np.c_[np.ones(raw_data.shape[0]), raw_data[:,0]]
y = np.c_[raw_data[:,1]]
m = y.shape[0]

theta = np.zeros((2,1)) # Initializing fitting parameters

# Importing the function
import plotData
plotData.plotData(X, y)

# Gradient descent settings
iterations = 1500
alpha = 0.01

import computeCost
J = computeCost.computeCost(X, y, theta)

import gradientDescent
theta, cost_J = gradientDescent.gradientDescent(X, y, theta, alpha, iterations)
print("Values of theta: %s" % theta)

# Verify that the cost function is monotonically decreasing with each iteration of gradient descent
plt.scatter(np.arange(iterations), cost_J)
plt.show()

plt.scatter(X[:,1], y, color='red', marker='x')
#plot the linear regression
lr_x = np.arange(5, np.amax(X))
lr_y = theta[0] + theta[1] * lr_x
plt.plot(lr_x, lr_y, label='Regression fit')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta) * 10000
predict2 = np.dot([1, 7], theta) * 10000
print(predict1, predict2)

### Part 4: Visualizing the Cost Function
# Creating the grid coordinates
B0 = np.linspace(-10,10,100)
B1 = np.linspace(-1,4,100)
xv, yv = np.meshgrid(B0,B1)
Z = np.zeros((B0.size, B1.size))
# Calculate cost based on each theta which are the grid coeffs x and y in each plot
for (i,j),v in np.ndenumerate(Z):
	Z[i,j] = computeCost.computeCost(X, y, theta=[[xv[i,j]], [yv[i,j]]])

fig = plt.figure(figsize=(10,5)) # The top level container for all of the plot elements
left = fig.add_subplot(1, 2, 1)
right = fig.add_subplot(1, 2, 2, projection='3d')

# Contour plot
countour_plot = left.contour(xv, yv, Z, np.logspace(-2, 3, 20), cmap=plt.cm.magma)
left.scatter(theta[0], theta[1], c='b', marker='x') # Plot the global minimum

# Surface plot
right.plot_surface(xv, yv, Z, rstride=1, cstride=1, cmap=plt.cm.inferno)
right.set_zlabel('cost j')
right.set_zlim(Z.min(), Z.max())
right.view_init(elev=5, azim=45)

# Label x and y with a nifty for loop
for i in fig.axes:
	i.set_xlabel(r'$\theta_0$', fontsize=10)
	i.set_ylabel(r'$\theta_1$', fontsize=10)

plt.show(fig)
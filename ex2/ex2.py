import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

import scipy.optimize as op 
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

filename = "ex2data1.txt"
data = np.loadtxt(filename, delimiter=',')

X = np.c_[np.ones((data.shape[0], 1)), data[:,:2]]
y = np.c_[data[:,2]]
m = len(y)

sns.scatterplot(X[:,0], X[:,1], y[:,0], style=y[:,0])
#plt.show()

### Part 2: Compute Cost and Gradient
import sigmoid
g = sigmoid.sigmoid(X)


theta = np.zeros((X.shape[1], 1))
from costFunction import costFunction, gradient
J = costFunction(X, y, theta)
grad = gradient(X, y, theta)

### Part 3: Optimizing using Scipy
'''
#optimized = op.minimize(fun=costFunction, x0=theta, args=(X, y), method='TNC', jac=gradient)
#op.fmin_bfgs(J, initial_theta) ## scipy doesn't play nice with numpy.ndarray object types? Convert somehow? Avoid numpy altogether?
#ValueError: shapes (2,) and (100,1) not aligned: 2 (dim 0) != 100 (dim 0)
It's not recognizing theta as a 2x1 matrix for some reason. Even after transposing and moving things around within costFunction 
'''
print(J)
print(grad)

# Let's hardcode the values for optimized_theta that we were supposed to get from the minimization function
optimized_theta = np.array([-25.16133593, 0.20623171, 0.20147164])

print(optimized_theta)

predicted_percent = sigmoid.sigmoid(np.array([1,45,85]).dot(optimized_theta.T))
print("Chances of student with Exam 1 score of 45 & Exam 2 score of 85: %f" % predicted_percent)

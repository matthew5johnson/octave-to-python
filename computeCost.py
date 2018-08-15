import numpy as np
import pandas as pd 


def computeCost(X, y, theta):
	m = y.size
	J = 0

	predictions = X.dot(theta)
	errors = predictions - y
	sqrErrors = np.square(errors)
	sumSqrErrors = np.sum(sqrErrors)
	J = (1./(2.*m))*sumSqrErrors

	return(J)


import matplotlib.pyplot as plt
import numpy as np 


def plotData(X, y):
	plt.scatter(X, y, linestyle='-', color='red', marker="x")
	plt.xlabel("Population of city in 10,000s")
	plt.ylabel("Profit in $10,000s")
	plt.show()
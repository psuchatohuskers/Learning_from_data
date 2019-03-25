import numpy as np
import matplotlib.pyplot as plt

def data_gen(N):
	"""The fuction takes an input N and retrun
		a generated numpy array fill with data sample
		from unif[-1,1] of size N-by-2 then add N-by-1
		bias term to result in N-by-3 matrix"""
	X_no_bias = np.random.uniform(-1,1,(N,2))
	X = np.hstack((np.ones((N,1)),X_no_bias))
	return X

def random_line():
	p0 = np.random.uniform(-1,1,2)
	p1 = np.random.uniform(-1,1,2)
	m = (p1[1]-p0[1])/(p1[0]-p0[0])
	b = p0[1] - (m*p0[0])
	f = np.array([b, m, -1])
	return f

def visualize_s(X,Y,f):
	"""This function take in input X, output Y and Target function f
		the plot the figure. This function can be used to check wheter or
		not output generation is correct"""
	x1 = np.linspace(-1,1, num=1000)
	x2 = f[1]*x1 + f[0]
	plt.plot(x1,x2)
	plt.scatter(x=X[:,1],y=X[:,2],c=Y,alpha=2.5,s=50)
	plt.xlim([-1,1])
	plt.ylim([-1,1])
	plt.show()

def train_set_gen(X,f):
	return np.sign(X.dot(f))

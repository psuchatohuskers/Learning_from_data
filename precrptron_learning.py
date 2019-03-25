# Preceptron Algorithm for Caltech's Learning from Data HW1
# Paloch Suchato
# 21/03/2019

import numpy as np
import random
import matplotlib.pyplot as plt
# set seed for testing consistency
def data_gen(N):
	"""The fuction takes an input N and retrun
		a generated numpy array fill with data sample
		from unif[-1,1] of size N-by-2 then add N-by-1
		bias term to result in N-by-3 matrix"""
	X_no_bias = np.random.uniform(-1,1,(N,2))
	X = np.hstack((np.ones((N,1)),X_no_bias))
	return X


def random_line():
	"""This function generate a line from two points
		drawn from unif[-1,1] by using normal equation
		(y - y1) = m(x - x1). This function return a numpy
		array of [b, m, -1] where b is the intercept,
		m is a slope and -1 is a slope of y when reaarage
		y = mx + b to cartesian plane form of b + mx - y = 0"""
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
	"""This function takes in inout X and target function f.
	This function generate output Y"""
	Y = X.dot(f)
	return np.sign(Y)

def preceptron(X,Y,epochs=1000):
	"""This fuction takes in input X, output Y and number of epoch
		This function train preceptron algorithm to fit the data
		and return the weight matrix for final hypothsis and number
		of iteration (epoch) that preceptron used in order to classified
		all points correctly
	"""
	w = np.array([0,0,0])
	for epoch in range(epochs):
		H = np.sign(X.dot(w))
		missclassified = np.where(Y != H)[0] #obtain a list of missclassified point
		if len(missclassified) == 0:
			break
		mc_sample = np.random.choice(missclassified) #pick one missclassified point
		w = w + (Y[mc_sample]*X[mc_sample])
	return w, epoch + 1


def visualize_g(X,Y,f,g):
	x1 = np.linspace(-1,1, num=1000)
	x2 = f[1]*x1 + f[0]
	x1h = np.linspace(-1,1, num=1000)
	x2h = g[1]*x1 + g[0]
	plt.plot(x1,x2)
	plt.plot(x1h,x2h)
	plt.scatter(x=X[:,1],y=X[:,2],c=Y,alpha=2.5,s=50)
	plt.xlim([-1,1])
	plt.ylim([-1,1])
	plt.show()


def p_error(f,g, N=10):
	test = data_gen(N)
	Y_target = np.sign(test.dot(f))
	Y_predict = np.sign(test.dot(g))
	predict_wrong = np.not_equal(Y_target,Y_predict).sum()
	pr_wrong = predict_wrong / float(len(Y_target))
	return pr_wrong


def simulate_preceptron(N,runs=1000):
	terminate_epoch = []
	p_wrong = []
	for run in range(runs):
		X = data_gen(N)
		f = random_line()
		Y = train_set_gen(X,f)
		g, epoch = preceptron(X,Y)
		terminate_epoch.append(epoch)
		p_wrong.append(p_error(f,g, N))
	mean_terminate_epoch = np.mean(terminate_epoch)
	mean_p_wrong = np.mean((p_wrong))
	return mean_terminate_epoch, mean_p_wrong

	


ter_epoch, p_wrong = simulate_preceptron(10)
print("(For N = 10) Iterations to converge: {}, P[f(x) != g(x)]: {}".format(ter_epoch,p_wrong))

ter_epoch, p_wrong = simulate_preceptron(100)
print("(For N = 100) Iterations to converge: {}, P[f(x) != g(x)]: {}".format(ter_epoch,p_wrong))

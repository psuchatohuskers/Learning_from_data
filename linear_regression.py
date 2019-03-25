import numpy as np
import matplotlib.pyplot as plt
from util import data_gen, random_line, visualize_s, train_set_gen

class LinearRegression(object):

    def fit(self,X,Y):
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
        return self.w

    def predict(self,X):
        H = np.sign(X.dot(self.w))
        return H

    def error(self,X,Y):
        H = np.sign(self.predict(X))
        return np.mean(Y != H)

def simulate_linreg(N_in,N_out,runs=1000):
    E_in = []
    E_out = []
    g = []
    for run in range(runs):
        linreg = LinearRegression()
        X_in  = data_gen(N_in)
        f = random_line()
        Y_in = train_set_gen(X_in,f)
        g_run = linreg.fit(X_in,Y_in)
        X_out = data_gen(N_out)
        Y_out = train_set_gen(X_out,f)
        g.append(g_run)
        E_run_in = linreg.error(X_in,Y_in)
        E_in.append(E_run_in)
        E_run_out = linreg.error(X_out, Y_out)
        E_out.append(E_run_out)
    error_in = np.mean(E_in)
    error_out = np.mean(E_out)
    return error_in,error_out, g

def preceptron(X,Y,g,epochs=1000):
    """This fuction takes in input X, output Y and number of epoch
        This function train preceptron algorithm to fit the data
        and return the weight matrix for final hypothsis and number
        of iteration (epoch) that preceptron used in order to classified
        all points correctly
    """
    w = g
    for epoch in range(epochs):
        H = np.sign(X.dot(w))
        missclassified = np.where(Y != H)[0] #obtain a list of missclassified point
        if len(missclassified) == 0:
            break
        mc_sample = np.random.choice(missclassified) #pick one missclassified point
        w = w + (Y[mc_sample]*X[mc_sample])
    return w, epoch + 1

def linreg_init_preceptron(N_in,runs=1000):
    terminate_epochs= []
    for run in range(runs):
        X = data_gen(N_in)
        f = random_line()
        Y = train_set_gen(X,f)
        linreg = LinearRegression()
        g_linreg = linreg.fit(X,Y)
        g, epoch = preceptron(X,Y,g_linreg)
        terminate_epochs.append(epoch)
    mean_terminate_epoch = np.mean(terminate_epochs)
    return mean_terminate_epoch



error_in,error_out, g, = simulate_linreg(100,1000,1000)
print("E_in is {} and E_out is {}".format(error_in,error_out))
print("terminate_epochs {}".format(linreg_init_preceptron(10)))

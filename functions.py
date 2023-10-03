import numpy as np

def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def logistic(x):
    return (1/(1+np.exp(-x)))

def logistic_prime(x):
    return logistic(x)*(1-logistic_prime(x))

def relu(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu_prime(x):
    return 1-relu(x)**2

def poslin(x):
    return x if x>=0 else 0

def poslin_prime(x):
    return 1 if x>=0 else 0

def purelin(x):
    return x

def purelin_prime(x):
    return 1

def hardlim(x):
    return 0 if x<0 else 1

def hardlim_prime(x):
    return 0 if x!=0 else null

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred,2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/(y_true.size);


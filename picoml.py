from ulab import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def mean_squared_error(y, t):
    y = np.array(y)
    t = np.array(t)
    return 0.5 * np.sum((y-t)**2)

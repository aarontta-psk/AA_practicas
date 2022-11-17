import numpy as np
import matplotlib.pyplot as plt

def gen_data(m, seed=1, scale=0.7):
    """ generate a data set based on a x^2 with added noise """
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,)) - 0.5)
    x_ideal  = x_train
    return x_train, y_train, x_ideal, y_ideal

def plot_data(x_ideal, y_train, y_ideal):
    """ plot the data, ideal red dotted line, train blue dots """
    plt.figure()
    plt.plot(x_ideal, y_ideal, 'r--', label='y_ideal')
    plt.plot(x_ideal, y_train, 'bo', label='train')
    plt.legend()
    plt.show()

def plot_data_comparing(x_ideal, y_ideal, x_train, y_train, x_pred, y_pred):
    """ plot the data, ideal red dotted line, train blue dots, pred blue line """
    plt.figure()
    plt.plot(x_train, y_train, 'bo', label='train', markersize=1)
    plt.plot(x_pred, y_pred, 'b-', label='predicted')
    plt.plot(x_ideal, y_ideal, 'r--', label='y_ideal')
    plt.legend()
    plt.show()

def plotCurvaAprendijaje(X, trainError, valError, linewidth=4):
    """ Plot the learning curve """
    plt.figure()
    plt.plot(X, trainError, 'b-', label='train', linewidth=linewidth)
    plt.plot(X, valError, '-', label='val', color='orange', linewidth=linewidth)
    plt.ylabel('Error')
    plt.xlabel('Number of Examples (m)')
    plt.legend()
    plt.show()
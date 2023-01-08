import nn as nn

import utils as ut

import numpy as np
import scipy.io as sc
import scipy.optimize as sciopt

def test_neural_network(theta1, theta2, X, y_onehot, lambda_):
    J = nn.cost(theta1, theta2, X, y_onehot, lambda_)
    print("Neural network cost:", J)
    print()
    ut.checkNNGradients(nn.backprop, lambda_)

def apply_nn(X, y_onehot, theta1, theta2, num_iters, alpha, lambda_, opt=True):
    # initialize thetas with a random factor
    epsilon = 0.12           # s[curr_l + 1]    s[curr_l] + 1    
    theta1 = np.random.random((theta1.shape[0], theta1.shape[1])) * (2 * epsilon) - epsilon
    theta2 = np.random.random((theta2.shape[0], theta2.shape[1])) * (2 * epsilon) - epsilon

    if opt == False:    # unoptimized
        _, tetha1, tetha2 = nn.gradient(theta1, theta2, X, y_onehot, num_iters, alpha, lambda_)
    else:               # optimized
        theta = np.concatenate([np.ravel(theta1), np.ravel(theta2)])
        res = sciopt.minimize(fun=nn.backprop_min, x0=theta, args=(X, y_onehot, theta1.shape, theta2.shape, lambda_), method='TNC', jac=True, options={'maxiter': num_iters})
        theta1 = np.reshape(res.x[:theta1.shape[0] * theta1.shape[1]], (theta1.shape[0], theta1.shape[1]))
        theta2 = np.reshape(res.x[theta1.shape[0] * theta1.shape[1]:], (theta2.shape[0], theta2.shape[1]))

    return theta1, theta2

def main():
    # initial data
    data = sc.loadmat('data/ex3data1.mat', squeeze_me=True)
    data_w = sc.loadmat('data/ex3weights.mat', squeeze_me=True)

    X, y = data['X'], data['y']
    theta1, theta2 = data_w['Theta1'], data_w['Theta2']

    alpha = 1
    lambda_ = 1
    num_iters = 1000
    n_labels = 10

    y_onehot = np.zeros((y.shape[0], n_labels))
    for i in range(y.shape[0]):
        y_onehot[i][y[i]] = 1
        
    # trying out the neural network on a small scale
    test_neural_network(theta1, theta2, X, y_onehot, lambda_)

    # use neural network gradient (by default, optimised)
    theta1, theta2 = apply_nn(X, y_onehot, theta1, theta2, num_iters, alpha, lambda_)

    # check accuracy
    predict = ut.predict(theta1, theta2, X)
    accuracy = np.sum(y == predict) / y.shape[0] * 100
    print('Neural network accuracy:', accuracy)

if __name__ == '__main__':
    main()
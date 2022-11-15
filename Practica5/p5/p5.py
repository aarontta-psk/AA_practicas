import nn as nn
import multi_class as mc
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import scipy.io as sc
import scipy.optimize as sciopt

def main():
    data = sc.loadmat('data/ex3data1.mat', squeeze_me=True)
    data_w = sc.loadmat('data/ex3weights.mat', squeeze_me=True)

    X, y = data['X'], data['y']
    theta1, theta2 = data_w['Theta1'], data_w['Theta2']


    input_layer_size = 3
    hidden_layer_size = 5
    n_labels = 10
    num_iters = 1000
    lambda_ = 1
    alpha = 1

    y_onehot = np.zeros((y.shape[0], n_labels))
    for i in range(y.shape[0]):
        y_onehot[i][y[i]] = 1

    J = nn.cost(theta1, theta2, X, y_onehot, lambda_)
    print(J)

    ut.checkNNGradients(nn.backprop, lambda_)

    epsilon = 0.12           
    theta1 = np.random.random((theta1.shape[0], theta1.shape[1])) * (2 * epsilon) - epsilon
    theta2 = np.random.random((theta2.shape[0], theta2.shape[1])) * (2 * epsilon) - epsilon

    # unoptimized
    # _, tetha1, tetha2 = nn.gradient(theta1, theta2, X, y_onehot, num_iters, alpha, lambda_)
    # optimized
    theta = np.concatenate([np.ravel(theta1), np.ravel(theta2)])
    res = sciopt.minimize(fun=nn.backprop_min, x0=theta, args=(X, y_onehot, theta1.shape, theta2.shape, lambda_), method='TNC', jac=True, options={'maxiter': num_iters})
    theta1 = np.reshape(res.x[:theta1.shape[0] * theta1.shape[1]], (theta1.shape[0], theta1.shape[1]))
    theta2 = np.reshape(res.x[theta1.shape[0] * theta1.shape[1]:], (theta2.shape[0], theta2.shape[1]))

    p = mc.predict(theta1, theta2, X)
    acc = np.sum(y == p) / y.shape[0] * 100
    print(acc)

    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
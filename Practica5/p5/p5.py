import nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc

def main():
    data = sc.loadmat('data/ex3data1.mat', squeeze_me=True)
    data_w = sc.loadmat('data/ex3weights.mat', squeeze_me=True)

    X, y = data['X'], data['y']
    theta1, theta2 = data_w['Theta1'], data_w['Theta2']

    n_labels = 10
    lambda_ = 1

    y_onehot = np.zeros((y.shape[0], n_labels))
    for i in range(y.shape[0]):
        y_onehot[i][y[i]] = 1

    J = nn.cost(theta1, theta2, X, y_onehot, lambda_)
    print(J)

    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
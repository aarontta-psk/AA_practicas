import multi_class as mc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc

def main():
    data = sc.loadmat('data/ex3data1.mat', squeeze_me=True)
    data_w = sc.loadmat('data/ex3weights.mat', squeeze_me=True)

    X, y = data['X'], data['y']
    theta1, theta2 = data_w['Theta1'], data_w['Theta2']

    n_labels = 10
    lambda_ = 0.001

    p = mc.predictOneVsAll(mc.oneVsAll(X, y, n_labels, lambda_), X)
    acc = np.sum(y == p) / y.shape[0] * 100

    p_w = mc.predict(theta1, theta2, X)
    acc_w = np.sum(y == p_w) / y.shape[0] * 100

    print(acc)
    print(acc_w)

    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
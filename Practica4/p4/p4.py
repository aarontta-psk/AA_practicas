# import public_tests as pt
import utils as ut 
import multi_class as mt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc

# def compute_g_descent(data):
#     x_train = data[:,:-1]
#     y_train = data[:,-1]

#     w_in = np.zeros(x_train.shape[1])
#     b_in = -8

#     alpha = 0.001
#     num_iters = 10000

#     w, b, costs = lr.gradient_descent(x_train, y_train, w_in, b_in, lr.compute_cost, lr.compute_gradient, alpha, num_iters)

#     plt.figure()
#     ut.plot_decision_boundary(w, b, x_train, y_train)
#     plt.savefig("./results/data.png")
#     plt.close('all')

#     return w, b, costs

# def compute_g_descent_reg(data):
#     new_data = np.column_stack((ut.map_feature(data[:,0], data[:,1]), data[:,-1]))

#     x_train = new_data[:,:-1]
#     y_train = new_data[:,-1]

#     w_in = np.zeros(x_train.shape[1])
#     b_in = 1

#     lambda_ = 0.01
#     alpha = 0.01
#     num_iters = 10000

#     w, b, costs = lr.gradient_descent(x_train, y_train, w_in, b_in, lr.compute_cost_reg, lr.compute_gradient_reg, alpha, num_iters, lambda_)

#     plt.figure()
#     ut.plot_decision_boundary(w, b, x_train, y_train)
#     plt.savefig("./results/data_reg.png")
#     plt.close('all')

#     return w, b, costs

def main():
    data = sc.loadmat('data/ex3data1.mat', squeeze_me=True)
    data_w = sc.loadmat('data/ex3weights.mat', squeeze_me=True)

    X, y = data['X'], data['y']
    theta1, theta2 = data_w['Theta1'], data_w['Theta2']

    n_labels = 10
    lambda_ = 0.001

    p = mt.predictOneVsAll(mt.oneVsAll(X, y, n_labels, lambda_), X)
    acc = np.sum(y == p) / y.shape[0] * 100

    p_w = mt.predict(theta1, theta2, X)
    acc_w = np.sum(y == p_w) / y.shape[0] * 100

    print(acc)
    print(acc_w)

    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
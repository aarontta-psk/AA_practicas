# import public_tests as pt
import utils as ut 
import logistic_reg as lr
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def compute_g_descent(data):
    x_train = data[:,:-1]
    y_train = data[:,-1]

    w_in = np.zeros(x_train.shape[1])
    b_in = -8

    alpha = 0.001
    num_iters = 10000

    w, b, costs = lr.gradient_descent(x_train, y_train, w_in, b_in, lr.compute_cost, lr.compute_gradient, alpha, num_iters)

    plt.figure()
    ut.plot_decision_boundary(w, b, x_train, y_train)
    plt.savefig("./results/data.png")
    plt.close('all')

    return w, b, costs

def compute_g_descent_reg(data):
    new_data = np.column_stack((ut.map_feature(data[:,0], data[:,1]), data[:,-1]))

    x_train = new_data[:,:-1]
    y_train = new_data[:,-1]

    w_in = np.zeros(x_train.shape[1])
    b_in = 1

    lambda_ = 0.01
    alpha = 0.01
    num_iters = 10000

    w, b, costs = lr.gradient_descent(x_train, y_train, w_in, b_in, lr.compute_cost_reg, lr.compute_gradient_reg, alpha, num_iters, lambda_)

    plt.figure()
    ut.plot_decision_boundary(w, b, x_train, y_train)
    plt.savefig("./results/data_reg.png")
    plt.close('all')

    return w, b, costs

def main():
    pt.sigmoid_test(lr.sigmoid)
    pt.compute_cost_test(lr.compute_cost)
    pt.compute_gradient_test(lr.compute_gradient)
    pt.predict_test(lr.predict)
    pt.compute_cost_reg_test(lr.compute_cost_reg)
    pt.compute_gradient_reg_test(lr.compute_gradient_reg)

    w, b, costs = compute_g_descent(ut.load_data())
    w, b, costs = compute_g_descent_reg(ut.load_data2())
    # dataN, mu, sigma = getNormalizedData()
    # testPrice(w, b, mu, sigma)

    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
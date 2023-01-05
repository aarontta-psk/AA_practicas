import logistic_reg as lgr

import public_tests as pt
import utils as ut 

import numpy as np

def compute_logistic_reg_gradient_descent(X_data, y_data, lambda_=None):
    # initial data and parameters
    X_train, y_train = X_data, y_data
    w_in, b_in = np.zeros(X_train.shape[1]), -8 if lambda_ is None else 1

    alpha = 0.001 if lambda_ is None else 0.01
    compute_cost, compute_grad = (lgr.compute_cost, lgr.compute_gradient) if lambda_ is None else (lgr.compute_cost_reg, lgr.compute_gradient_reg)
    num_iters = 10000

    # we obtain w and b here
    w, b, costs = lgr.gradient_descent(X_train, y_train, w_in, b_in, compute_cost, compute_grad, alpha, num_iters, lambda_)

    return w, b, costs

def main():
    # initial tests for logistic reg functions
    pt.sigmoid_test(lgr.sigmoid)
    pt.compute_cost_test(lgr.compute_cost)
    pt.compute_gradient_test(lgr.compute_gradient)
    pt.compute_cost_reg_test(lgr.compute_cost_reg)
    pt.compute_gradient_reg_test(lgr.compute_gradient_reg)
    pt.predict_test(lgr.predict)

    # parameters acquisition and visualization 
    X_data, y_data = ut.load_data('./data/ex2data1.txt')
    w, b, _ = compute_logistic_reg_gradient_descent(X_data, y_data)
    ut.plot_logistic_reg("./results/logistic_reg_fit.png", X_data, y_data, w, b)

    # parameters acquisition and visualization (regularized)
    X_data, y_data = ut.load_data('./data/ex2data2.txt')
    data = ut.map_data_features(X_data, y_data) # map features 
    w, b, _ = compute_logistic_reg_gradient_descent(data[:,:-1], data[:,-1], 0.1)
    ut.plot_logistic_reg("./results/logistic_reg_fit_regularized.png", data[:,:-1], data[:,-1], w, b)

if __name__ == '__main__':
    main()
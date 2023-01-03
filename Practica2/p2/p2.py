import multi_linear_reg as mlr

import public_tests as pt
import utils as ut

import numpy as np

def compute_multi_linear_reg_descent(data):
    # initial data and parameters
    x_train, y_train = data[:,:4], data[:,4]
    w_in, b_in = np.zeros(x_train.shape[1]), 0

    alpha = 0.01
    num_iters = 1500

    # we obtain w and b here
    w, b, costs = mlr.gradient_descent(x_train, y_train, w_in, b_in, mlr.compute_cost, mlr.compute_gradient, alpha, num_iters)

    return w, b, costs

def test_parameters(w, b, mu, sigma):
    # set data
    features = np.array([1200, 3, 1, 40])
    norm_features = (features - mu) / sigma

    # should be around $318709
    price = (np.dot(w, norm_features) + b) * 1000
    print("Price for a house with 1200 sqft, 3 bedrooms, 1 foor, 40 years old: ${}".format(price))
    
def main():
    # initial tests for multivariable linear reg functions
    pt.compute_cost_test(mlr.compute_cost)
    pt.compute_gradient_test(mlr.compute_gradient)

    # parameters acquisition and TODO visualization 
    X_data, y_data = ut.load_data('./data/houses.txt')
    norm_data, mu, sigma = ut.normalize_data(X_data, y_data)
    w, b, _ = compute_multi_linear_reg_descent(norm_data)
    ut.plot_multi_linear_reg('./results/multi_linear_reg_fit.png', w, b, X_data, y_data, norm_data)

    # test results
    test_parameters(w, b, mu, sigma)

if __name__ == '__main__':
    main()
import public_tests as pt
import utils as ut
import multi_linear_reg as mlr

import matplotlib as plt
import numpy as np

def getNormalizedData():
    data = ut.load_data()
    dataN, mu, sigma = mlr.zscore_normalize_features(data[:, :4])
    # Apend last column of data to dataN
    dataN = np.append(dataN, data[:, 4].reshape(-1, 1), axis=1)
    return [dataN, mu, sigma]

def compute_g_descent(data):
    x_train = data[:,:4]
    y_train = data[:,4]

    w_in = np.zeros(x_train.shape[1])
    b_in = 0

    alpha = 0.01
    num_iters = 1500

    w, b, costs = mlr.gradient_descent(x_train, y_train, w_in, b_in, mlr.compute_cost, mlr.compute_gradient, alpha, num_iters)

    return w, b, costs

def normalizeData(mu, sigma, data):
    return (data - mu) / sigma

def estimatePrice(w, b, X):
    return (np.dot(w, X) + b) * 1000

def testPrice(w, b, mu, sigma):
    feature = normalizeData(mu, sigma, np.array([1200, 3, 1, 40]))
    price = estimatePrice(w, b, feature)
    print("Price for a house with 1200 sqft, 3 bedrooms, 1 foor, 40 years old: ${}".format(price))
    # $318709
    
def main():
    pt.compute_cost_test(mlr.compute_cost)
    pt.compute_gradient_test(mlr.compute_gradient)

    dataN, mu, sigma = getNormalizedData()
    w, b, _ = compute_g_descent(dataN)
    testPrice(w, b, mu, sigma)

    # print("\033[0m", end = '')

if __name__ == '__main__':
    main()
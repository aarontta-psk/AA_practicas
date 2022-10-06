import public_tests as pt
import utils as ut 
import logistic_reg as lr
import numpy as np

def compute_g_descent(data):
    x_train = data[:,:-1]
    y_train = data[:,-1]

    w_in = np.zeros(x_train.shape[0])
    b_in = -8

    alpha = 0.01
    num_iters = 10000

    w, b, costs = lr.gradient_descent(x_train, y_train, w_in, b_in, lr.compute_cost, lr.compute_gradient, alpha, num_iters)

    ut.plot_decision_boundary(w, b, x_train, y_train)

    return w, b, costs

def main():
    pt.sigmoid_test(lr.sigmoid)
    pt.compute_cost_test(lr.compute_cost)
    pt.compute_gradient_test(lr.compute_gradient)
    pt.predict_test(lr.predict)

    w, b, costs = compute_g_descent(ut.load_data())
    # dataN, mu, sigma = getNormalizedData()
    # testPrice(w, b, mu, sigma)

    # print("\033[0m", end = '')

if __name__ == '__main__':
    main()
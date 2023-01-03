import public_tests as pt
import utils as ut

import linear_reg as lr

def compute_linear_reg_gradient_descent(data):
    # initial data and parameters
    x, y = data[0], data[1]
    w_in = b_in = 0

    alpha = 0.01
    num_iters = 1500

    # we obtain the best w and b here
    w, b, costs = lr.gradient_descent(x, y, w_in, b_in, lr.compute_cost, lr.compute_gradient, alpha, num_iters)

    return w, b, costs

def main():
    # initial tests for linear reg functions
    pt.compute_cost_test(lr.compute_cost)
    pt.compute_gradient_test(lr.compute_gradient)
    
    # parameters acquisition and visualization
    data = ut.load_data('./data/ex1data1.txt')
    w, b, _ = compute_linear_reg_gradient_descent(data)
    ut.plot_linear_reg('./results/linear_reg_fit.png', w, b, data)

if __name__ == '__main__':
    main()
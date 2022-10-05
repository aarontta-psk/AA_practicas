import public_tests as pt
import utils as ut
import linear_reg as lr

import matplotlib.pyplot as plt

def compute_g_descent(data):
    x = data[0]
    y = data[1]

    w_in = 0
    b_in = 0

    alpha = 0.01
    num_iters = 1500

    w, b, costs = lr.gradient_descent(x, y, w_in, b_in, lr.compute_cost, lr.compute_gradient, alpha, num_iters)

    return w, b, costs

def render_data():
    data = ut.load_data()

    plt.figure()
    
    plt.title("Profits vs. Population")
    plt.ylabel("Profits (10,000$)")
    plt.xlabel("Population of City (10,000s)")
    
    plt.scatter(data[0], data[1], c='red', marker='x')
    
    # subplot = plt.subplot()
    # subplot.spines.right.set_visible(False)
    # subplot.spines.top.set_visible(False)

def plot():
    render_data()

    plt.savefig("./results/data.png")
    plt.close('all')

def plot_linear(w, b):
    render_data()
    
    data = ut.load_data()
    x = data[0]

    plt.plot(x, w * x + b, c='blue')

    plt.savefig("./results/linear_fit.png")
    plt.close('all')

def main():
    pt.compute_cost_test(lr.compute_cost)
    pt.compute_gradient_test(lr.compute_gradient)
    
    w, b, costs = compute_g_descent(ut.load_data())

    plot()
    plot_linear(w, b)

if __name__ == '__main__':
    main()
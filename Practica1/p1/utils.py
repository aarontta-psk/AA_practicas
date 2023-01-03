import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    X, y = data[:,0], data[:,1]
    return X, y

def plot_linear_reg_data(data):
    plt.figure()
    
    plt.title("Profits vs. Population")
    plt.ylabel("Profits (10,000$)")
    plt.xlabel("Population of City (10,000s)")
    
    plt.scatter(data[0], data[1], c='red', marker='x')

    # hides graph limits
    # subplot = plt.subplot()
    # subplot.spines.right.set_visible(False)
    # subplot.spines.top.set_visible(False)

    plt.savefig("./results/linear_reg_data.png")

def plot_linear_reg(path, w, b, data):
    plot_linear_reg_data(data)

    X = data[0]
    plt.plot(X, w * X + b, c='blue')

    plt.savefig(path)
    plt.close('all')
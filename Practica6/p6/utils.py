import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scp

def gen_data(m, seed=1, scale=0.7):
    # generate a data set based on a x^2 with added noise
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal

def plot_train(X, y, x_ideal, y_ideal, X_predict, y_predict, fileName, title):
    plt.figure()

    plt.title(title)
    plt.scatter(X, y, c='blue', label='train')
    plt.plot(x_ideal, y_ideal, c='red', label='y_ideal', linestyle='dashed')

    y_sorted = [y for _,y in sorted(zip(X_predict, y_predict))]
    x_sorted = np.sort(X_predict, axis=None)

    x_sorted_soften = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
    spline = scp.make_interp_spline(x_sorted, y_sorted)
    y_sorted_soften = spline(x_sorted_soften)

    plt.plot(x_sorted_soften, y_sorted_soften, c='black', label='predict', linestyle='dashed', linewidth=1)

    plt.legend()
    plt.savefig(fileName)
    plt.close()

def plot_error(num_data, error_data_train, error_data_cv, fileName, linewidth = 4):
    plt.figure()

    plt.title("Error graph")
    plt.xlabel("Number cases")
    plt.ylabel("Error")

    plt.plot(num_data, error_data_train, c='cyan', label='error_train', linewidth=3)
    plt.plot(num_data, error_data_cv,c='orange', label='error_cv', linewidth=3)

    plt.legend()
    plt.savefig(fileName) # plt.show()
    plt.close()
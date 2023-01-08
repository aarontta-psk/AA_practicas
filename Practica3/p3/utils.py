import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    X, y = data[:,:-1], data[:,-1]
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def map_feature(X1, X2, degree=6):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

def map_data_features(X_data, y_data):
    return np.column_stack((map_feature(X_data[:,0], X_data[:,1]), y_data))

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)

def plot_decision_boundary(w, b, X, y):
    # Credit to dibgerge on Github for this plotting code
    plot_data(X[:, 0:2], y)

    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0
        plt.contour(u, v, z, levels=[0.5], colors="g")

def plot_logistic_reg(path, title, X_train, y_train, w, b):
    plt.figure()

    plt.title(title)
    plot_decision_boundary(w, b, X_train, y_train)

    plt.savefig(path)
    plt.close('all')
import multi_linear_reg as mlr
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X, y = data[:,:-1], data[:,-1]
    return X, y

def normalize_data(X_data, y_data):
    # normalize the data and obtain paramters for future usage
    norm_data, mu, sigma = mlr.zscore_normalize_features(X_data)
    # append last column of data (y) to the normalized data
    norm_data = np.append(norm_data, y_data.reshape(-1, 1), axis=1)
    
    return norm_data, mu, sigma

def plot_multi_linear_reg_data(X_train, y_train):
    X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

    plt.figure()

    fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train)
        ax[i].set_xlabel(X_features[i])
    
    ax[0].set_ylabel("Price (1000's)")

    plt.savefig("./results/multi_linear_reg_data.png")

    return ax

def plot_multi_linear_reg(path, w, b, X_data, y_data, norm_data):
    ax = plot_multi_linear_reg_data(X_data, y_data)
    
    y_predict = (norm_data[:,:-1] @ w) + b

    for i in range(len(ax)):
        ax[i].scatter(X_data[:, i], y_predict, c='orange')

    plt.savefig(path)
    plt.close('all')
import numpy as np
import utils as ut
import sklearn.model_selection as skm
import sklearn.preprocessing as skp
import sklearn.linear_model as skl
import matplotlib.pyplot as plt

def gen_data(m, seed=1, scale=0.7):
    # generate a data set based on a x^2 with added noise
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal

def main():
    X, y, x_ideal, y_ideal = gen_data(64)
    m = X.shape[0]

    plt.figure()
    plt.scatter(X, y, c='blue', label='train')
    plt.plot(x_ideal, y_ideal, c='red', label='y_ideal')

    X = X[:, None]

    X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.33, random_state=1)

    poly = skp.PolynomialFeatures(degree=15, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    std = skp.StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)

    linear = skl.LinearRegression()
    linear.fit(X_train, y_train)

    error_train = np.sum((linear.predict(X_train) - y_train)**2) / (2*m)
    error_test = np.sum((linear.predict(X_test) - y_test)**2) / (2*m)

    print(error_train)
    print(error_test)

    # X1 = np.sort(X)
    # Y1 = np.sort(Y)

    # plt.plot(X1, Y1, c='yellow', label='predict')
    plt.legend()
    plt.savefig('graph.pdf')

    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
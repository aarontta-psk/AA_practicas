import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scp

import sklearn.model_selection as skm
import sklearn.preprocessing as skp
import sklearn.linear_model as skl

def gen_data(m, seed=1, scale=0.7):
    # generate a data set based on a x^2 with added noise
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal

def train(x_train, y_train, degree=15, lambda_=0):
    poly = skp.PolynomialFeatures(degree=degree, include_bias=False)
    scalar = skp.StandardScaler()
    model = skl.Ridge(alpha=lambda_) # model = skl.LinearRegression()

    x_train = poly.fit_transform(x_train)
    x_train = scalar.fit_transform(x_train)

    model.fit(x_train, y_train)

    return poly, scalar, model

def calc_error(poly, scalar, model, x, y):
    x_test = poly.transform(x)
    x_test = scalar.transform(x_test)

    y_predict = model.predict(x_test)

    return np.sum((y_predict - y)**2) / (2 * y.shape[0]), x_test, y_predict[:, None]

def plot_train(X, y, x_ideal, y_ideal, X_predict, y_predict, fileName):
    plt.figure()

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

    # print(num_data)
    # print(error_data_train)
    # print(error_data_cv)

    # plt.figure()

    # plt.xlabel("Num casos")
    # plt.ylabel("Error")

    # plt.plot(num_data, error_data_train, c='cyan', label='error_train', linewidth=3)
    # plt.plot(num_data, error_data_cv,c='orange', label='error_cv', linewidth=3)

    # plt.legend()
    # # plt.savefig(fileName)
    # plt.show()
    # plt.close()
    plt.figure()
    plt.plot(num_data, error_data_train, 'b-', label='train', linewidth=linewidth)
    plt.plot(num_data, error_data_cv, '-', label='val', color='orange', linewidth=linewidth)
    plt.ylabel('Error')
    plt.xlabel('Number of Examples (m)')
    plt.legend()
    plt.show()

def test_error():
    num_data = np.linspace(50, 1000, 20)
    error_data_train = np.zeros(len(num_data))
    error_data_cv = np.zeros(len(num_data))

    for value in range(len(num_data)):
        X, y, x_ideal, y_ideal = gen_data(int(num_data[value]))
        X = X[:, None]

        x_train, x_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.4, random_state=1)
        x_test, x_cv, y_test, y_cv = skm.train_test_split(x_test, y_test, test_size=0.5, random_state=1)

        degree = 16
        poly, scalar, model = train(x_train, y_train, degree)

        error_data_train[value], _, _ = calc_error(poly, scalar, model, x_train, y_train)
        error_data_cv[value], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    plot_error(num_data, error_data_train, error_data_cv, './p6/docs/test_error.pdf')

def test_hiper_parameters(X, y, x_ideal, y_ideal):
    x_train, x_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.4, random_state=1)
    x_test, x_cv, y_test, y_cv= skm.train_test_split(x_test, y_test, test_size=0.5, random_state=1)

    degrees = 15
    lambdas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900])
    error_data = np.zeros((len(lambdas), degrees))
    for temp_lambda in range(len(lambdas)):
        for temp_degree in range(degrees):
            poly, scalar, model = train(x_train, y_train, temp_degree + 1, lambdas[temp_lambda])

            error_data[temp_lambda, temp_degree], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    min = np.argmin(error_data)
    print("Min index:", min)

    lambda_id = int(min/len(lambdas))
    proper_lambda = lambdas[lambda_id]  # MASSIVE PROBLEM, COLOSSAL EVEN
    proper_degree = min - (len(lambdas) * lambda_id) + 1 
    print("Proper degree:", proper_degree)
    print("Proper lambda:", proper_lambda)
    print("Error degree&lambda:", error_data)

    poly, scalar, model = train(x_train, y_train, proper_degree, proper_lambda)

    error, _, y_predict = calc_error(poly, scalar, model, x_test, y_test)
    print("Error test:", error)

    plot_train(X, y, x_ideal, y_ideal, x_test, y_predict, './p6/docs/test_hiper_parameters.pdf')

def test_lambda(X, y, x_ideal, y_ideal):
    x_train, x_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.4, random_state=1)
    x_cv, x_test, y_cv, y_test = skm.train_test_split(x_test, y_test, test_size=0.5, random_state=1)

    degree = 15
    lambdas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900])
    error_data = np.zeros(len(lambdas))
    for temp_lambda in range(len(lambdas)):
        poly, scalar, model = train(x_train, y_train, degree, lambdas[temp_lambda])

        error_data[temp_lambda], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    proper_lambda = lambdas[np.argmin(error_data)] # MASSIVE PROBLEM, COLOSSAL EVEN
    print("Proper lambda:", proper_lambda)
    # print("Error lambda:", error_data)

    poly, scalar, model = train(x_train, y_train, degree, proper_lambda)

    _, _, y_predict = calc_error(poly, scalar, model, x_test, y_test)

    plot_train(X, y, x_ideal, y_ideal, x_test, y_predict, './p6/docs/test_lambda.pdf')

def test_degree(X, y, x_ideal, y_ideal):
    x_train, x_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.4, random_state=1)
    x_cv, x_test, y_cv, y_test = skm.train_test_split(x_test, y_test, test_size=0.5, random_state=1)

    error_data = np.zeros(10) # degree from 1 to 10 [(0,9) + 1]
    for temp_degree in range(len(error_data)):
        poly, scalar, model = train(x_train, y_train, temp_degree + 1)

        error_data[temp_degree], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    proper_degree = np.argmin(error_data) + 1
    print("Proper degree:", proper_degree)
    # print("Error degree:", error_data)

    poly, scalar, model = train(x_train, y_train, proper_degree)

    _, _, y_predict = calc_error(poly, scalar, model, x_test, y_test)

    plot_train(X, y, x_ideal, y_ideal, x_test, y_predict, './p6/docs/test_degree.pdf')


def test_start(X, y, x_ideal, y_ideal):
    x_train, x_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.33, random_state=1)

    poly, scalar, model = train(x_train, y_train)

    error_train, _, y_train_predict = calc_error(poly, scalar, model , x_train, y_train)
    error_test, _, _ = calc_error(poly, scalar, model, x_test, y_test)

    print("Error train:", error_train)
    print("Error test:", error_test)

    plot_train(X, y, x_ideal, y_ideal, x_train, y_train_predict, './p6/docs/test_start.pdf')


def main():
    X, y, x_ideal, y_ideal = gen_data(64)
    X = X[:, None]

    test_start(X, y, x_ideal, y_ideal)
    test_degree(X, y, x_ideal, y_ideal)
    test_lambda(X, y, x_ideal, y_ideal)

    X, y, x_ideal, y_ideal = gen_data(750)
    X = X[:, None]
    test_hiper_parameters(X, y, x_ideal, y_ideal)

    # test_error()

    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
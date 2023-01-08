import utils as ut

import numpy as np

import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import sklearn.linear_model as skl

def calc_error(poly, scalar, model, x, y):
    x_test = poly.transform(x)
    x_test = scalar.transform(x_test)

    y_predict = model.predict(x_test)

    return np.sum((y_predict - y)**2) / (2 * y.shape[0]), x_test, y_predict[:, None]

def train(x_train, y_train, degree=15, lambda_=0):
    poly = skp.PolynomialFeatures(degree=degree, include_bias=False)
    scalar = skp.StandardScaler()
    model = skl.Ridge(alpha=lambda_) # model = skl.LinearRegression()

    x_train = poly.fit_transform(x_train)
    x_train = scalar.fit_transform(x_train)

    model.fit(x_train, y_train)

    return poly, scalar, model

# hiper parameters test
def test_error():
    num_data = np.linspace(50, 1000, 20)
    error_data_train = np.zeros(len(num_data))
    error_data_cv = np.zeros(len(num_data))

    for value in range(len(num_data)):
        X, y, x_ideal, y_ideal = ut.gen_data(int(num_data[value]))
        X = X[:, None]

        x_train, x_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.4, random_state=1)
        x_test, x_cv, y_test, y_cv = skms.train_test_split(x_test, y_test, test_size=0.5, random_state=1)

        degree = 16
        poly, scalar, model = train(x_train, y_train, degree)

        error_data_train[value], _, _ = calc_error(poly, scalar, model, x_train, y_train)
        error_data_cv[value], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    ut.plot_error(num_data[1:], error_data_train[1:], error_data_cv[1:], './results/test_error.png')

# hiper parameters test
def test_hiper_parameters(X, y, x_ideal, y_ideal):
    x_train, x_, y_train, y_ = skms.train_test_split(X, y, test_size=0.4, random_state=1)
    x_test, x_cv, y_test, y_cv = skms.train_test_split(x_, y_, test_size=0.5, random_state=1)

    degrees = 15
    lambdas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900])
    error_data = np.zeros((len(lambdas), degrees))
    for temp_lambda in range(len(lambdas)):
        for temp_degree in range(degrees):
            poly, scalar, model = train(x_train, y_train, temp_degree + 1, lambdas[temp_lambda])

            error_data[temp_lambda, temp_degree], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    min = np.argmin(error_data)
    lambda_id = int(min/len(lambdas))
    proper_lambda = lambdas[lambda_id]
    proper_degree = min - (len(lambdas) * lambda_id) + 1 
    
    print("--HIPER PARAMETERS TEST--")
    print("Proper degree:", proper_degree)
    print("Proper lambda:", proper_lambda)

    poly, scalar, model = train(x_train, y_train, proper_degree, proper_lambda)

    error, _, y_predict = calc_error(poly, scalar, model, x_test, y_test)

    print("Error test:", error)
    print()

    ut.plot_train(X, y, x_ideal, y_ideal, x_test, y_predict, './results/test_hiper_parameters.png', "Test hiper-parameters")

# lambda test
def test_lambda(X, y, x_ideal, y_ideal):
    x_train, x_, y_train, y_ = skms.train_test_split(X, y, test_size=0.4, random_state=1)
    x_cv, x_test, y_cv, y_test = skms.train_test_split(x_, y_, test_size=0.5, random_state=1)

    degree = 15
    lambdas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900])
    error_data = np.zeros(len(lambdas))
    for temp_lambda in range(len(lambdas)):
        poly, scalar, model = train(x_train, y_train, degree, lambdas[temp_lambda])

        error_data[temp_lambda], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    proper_lambda = lambdas[np.argmin(error_data)]
    
    print("--LAMBDA TEST--")
    print("Proper lambda:", proper_lambda)

    poly, scalar, model = train(x_train, y_train, degree, proper_lambda)

    error_test, _, y_predict = calc_error(poly, scalar, model, x_test, y_test)

    print("Error lambda:", error_test)
    print()
    
    ut.plot_train(X, y, x_ideal, y_ideal, x_test, y_predict, './results/test_lambda.png', "Test lambda")

# degree test
def test_degree(X, y, x_ideal, y_ideal):
    x_train, x_, y_train, y_ = skms.train_test_split(X, y, test_size=0.4, random_state=1)
    x_cv, x_test, y_cv, y_test = skms.train_test_split(x_, y_, test_size=0.5, random_state=1)

    error_data = np.zeros(10) # degree from 1 to 10 [(0,9) + 1]
    for temp_degree in range(len(error_data)):
        poly, scalar, model = train(x_train, y_train, temp_degree + 1)

        error_data[temp_degree], _, _ = calc_error(poly, scalar, model, x_cv, y_cv)

    proper_degree = np.argmin(error_data) + 1

    print("--DEGREE TEST--")
    print("Proper degree:", proper_degree)

    poly, scalar, model = train(x_train, y_train, proper_degree)

    error_test, _, y_predict = calc_error(poly, scalar, model, x_test, y_test)
    
    print("Error degree:", error_test)
    print()

    ut.plot_train(X, y, x_ideal, y_ideal, x_test, y_predict, './results/test_degree.png', "Test degree")

# first test
def test_overadjustment(X, y, x_ideal, y_ideal):
    x_train, x_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.33, random_state=1)

    poly, scalar, model = train(x_train, y_train)

    error_train, _, y_train_predict = calc_error(poly, scalar, model , x_train, y_train)
    error_test, _, _ = calc_error(poly, scalar, model, x_test, y_test)

    print("--OVERADJUSTMENT TEST--")
    print("Error train:", error_train)
    print("Error test:", error_test)
    print()

    ut.plot_train(X, y, x_ideal, y_ideal, x_train, y_train_predict, './results/test_overadjustment.png', "Test overadjustment")

def main():
    X, y, x_ideal, y_ideal = ut.gen_data(64)
    X = X[:, None]

    test_overadjustment(X, y, x_ideal, y_ideal)
    test_degree(X, y, x_ideal, y_ideal)
    test_lambda(X, y, x_ideal, y_ideal)

    X, y, x_ideal, y_ideal = ut.gen_data(750)
    X = X[:, None]
    test_hiper_parameters(X, y, x_ideal, y_ideal)

    test_error()

if __name__ == '__main__':
    main()
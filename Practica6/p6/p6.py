import numpy as np
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

def main():
    X, y, x_ideal, y_ideal = gen_data(65)

    X = X[:, None]
    m = X.shape[0]

    X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.33, random_state=1)

    poly = skp.PolynomialFeatures(degree=15, include_bias=False)
    poly.fit_transform(X_train)
    poly.fit_transform(X_test)

    std = skp.StandardScaler()
    std.fit_transform(X_train)
    std.fit_transform(X_test)

    linear = skl.LinearRegression()
    linear.fit(X_train)

    error_train = (linear.predict(X_train) - y)**2 / (2*m)
    error_test = (linear.predict(X_test) - y)**2 / (2*m)

    print(error_train)
    print(error_test)


    print("\033[0m", end = '')

if __name__ == '__main__':
    main()
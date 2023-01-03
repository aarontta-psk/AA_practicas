import multi_class as mc

import numpy as np
import scipy.io as sc

def main():
    # data & parameters
    data = sc.loadmat('data/ex3data1.mat', squeeze_me=True)
    data_weights = sc.loadmat('data/ex3weights.mat', squeeze_me=True)

    X, y = data['X'], data['y']
    theta1, theta2 = data_weights['Theta1'], data_weights['Theta2']

    n_labels = 10
    lambda_ = 0.001

    # logistic regression oneVsAll prediction
    predict = mc.predictOneVsAll(mc.oneVsAll(X, y, n_labels, lambda_), X)
    accuracy = np.sum(y == predict) / y.shape[0] * 100
    print("Accuracy oneVsAll:", accuracy)

    # pre trained neural network prediction
    predict_weights = mc.predict(theta1, theta2, X)
    accuracy = np.sum(y == predict_weights) / y.shape[0] * 100
    print("Accuracy Neural Network:", accuracy)

if __name__ == '__main__':
    main()
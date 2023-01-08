import svm 

import utils as ut
import numpy as np

import sklearn.svm as skl_svm
import sklearn.metrics as skm

def test_svm(X, y, res_path, title, kernel_type, c, sigma):
    svm_ = svm.svm(X, y, None, None, kernel_type, c, sigma)
    ut.plot_svm(res_path, title, X, y, svm_)

def apply_svm(X, y, X_val, y_val, kernel_type, c, sigma):
    svm_, _, accuracy = svm.svm(X, y, X_val, y_val, kernel_type, c, sigma)
    return svm_, accuracy * 100

def get_best_svm_parameters(X, y, X_val, y_val):
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    accuracies, svm_ = [], []
    for c_value in values:
        for sigma_value in values:
            new_svm, new_accuracy = apply_svm(X, y, X_val, y_val, 'rbf', c_value, sigma_value)
            
            accuracies.append(new_accuracy)
            svm_.append((new_svm, c_value, sigma_value))

    best_accuracy = np.argmax(accuracies)
    svm_, c, sigma = svm_[best_accuracy]

    print("C:", c)
    print("Sigma:", sigma)
    print('Accuracy:', accuracies[best_accuracy])

    ut.plot_svm('./results/graph_c_' + str(c) + '_sigma_' + str(sigma) + '.png', 'c_' + str(c) + '_sigma_' + str(sigma), X, y, svm_)

def main():
    # testing c values
    X, y = ut.obtain_data('./data/ex6data1.mat')
    test_svm(X, y, './results/graph_c1.png', 'c_1', 'linear', 1.0, 1.0)
    test_svm(X, y, './results/graph_c2.png', 'c_10', 'linear', 10.0, 1.0)

    # testing gaussian kernel
    X, y = ut.obtain_data('./data/ex6data2.mat')
    test_svm(X, y, './results/graph_sigma.png', 'sigma_0.1', 'rbf', 1.0, 0.1)

    # picking best parameters
    X, y, X_val, y_val = ut.obtain_data('./data/ex6data3.mat')
    get_best_svm_parameters(X, y, X_val, y_val)

if __name__ == '__main__':
    main()
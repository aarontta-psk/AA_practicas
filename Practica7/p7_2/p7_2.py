import logistic_reg as lgr
import svm as svm
import nn as nn

import utils as ut

import numpy as np
import sklearn.metrics as skme

def obtain_svm(x_train, y_train, x_val, y_val):
    # initial values
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
    # check for best parameters
    svm_ = None
    best_acc = -1 
    for c in values:
        for sigma in values:
            new_svm, _, accuracy = svm.svm(x_train, y_train, x_val, y_val, 'rbf', c, sigma)
            
            if (best_acc < accuracy):
                best_acc = accuracy
                svm_ = new_svm
                
    return svm_

def obtain_reg_log(x_train, y_train, x_val, y_val):
    # initial values
    values = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500])
    w_in, b_in = np.zeros(x_train.shape[1]), 1
    num_iters = 1000
    alpha = 0.01

    # check for best parameters
    lambda_ = 0
    best_acc = -1 
    for temp_lambda in values:
        new_w, new_b, _ = lgr.gradient_descent(x_train, y_train, w_in, b_in, lgr.compute_cost_reg, lgr.compute_gradient_reg, alpha, num_iters, temp_lambda)

        predict = lgr.predict(x_val, new_w, new_b)
        accuracy = np.sum(y_val == predict) / y_val.shape[0]

        if(best_acc < accuracy):
            best_acc = accuracy
            lambda_ = temp_lambda

    # redo best regression with more iterations
    num_iters = 10000
    w, b, _ = lgr.gradient_descent(x_train, y_train, w_in, b_in, lgr.compute_cost_reg, lgr.compute_gradient_reg, alpha, num_iters, lambda_)

    return w, b

def obtain_neural_network(x_train, y_train, x_val, y_val):
    # initial values
    # values = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500])
    values = [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 50]

    output_labels = 2               # spam or not spam
    input_labels = len(x_train[0])  # list words email
    hidden_labels = 25              # because I can

    theta1, theta2 = nn.random_thetas(input_labels, hidden_labels, output_labels)
    y_onehot = nn.onehot(y_train, output_labels)
    num_iters = 100

    # check for best parameters
    lambda_ = 0
    best_acc = -1 
    for temp_lambda in values:
        theta_1, theta_2 = nn.minimize(x_train, y_onehot, theta1, theta2, num_iters, temp_lambda)
        
        predict = ut.predict(theta_1, theta_2, x_val)
        accuracy = np.sum(y_val == predict) / y_val.shape[0]

        print('Lambda: ', temp_lambda)

        if(best_acc < accuracy):
            best_acc = accuracy
            lambda_ = temp_lambda

    # redo best neural network with more iterations
    num_iters = 1000
    theta1, theta2 = nn.minimize(x_train, y_onehot, theta1, theta2, num_iters, lambda_)

    return theta1, theta2

def check_spam_svm(x_train, y_train, x_val, y_val, x_test, y_test):
    svm = obtain_svm(x_train, y_train, x_val, y_val)
    accuracy = skme.accuracy_score(y_test, svm.predict(x_test))
    print("SVM best accuracy: ", accuracy)

def check_spam_reg_log(x_train, y_train, x_val, y_val, x_test, y_test):
    w, b = obtain_reg_log(x_train, y_train, x_val, y_val)
    predict = lgr.predict(x_test, w, b)
    accuracy = np.sum(y_test == predict) / y_test.shape[0]
    print("Logistic regression best accuracy: ", accuracy)
    
def check_spam_neural_network(x_train, y_train, x_val, y_val, x_test, y_test):
    theta1, theta2 = obtain_neural_network(x_train, y_train, x_val, y_val)
    p = ut.predict(theta1, theta2, x_test)
    accuracy = np.sum(y_test == p) / y_test.shape[0]
    print("Neural network best accuracy: ", accuracy)

def main(training_sys):
    x_train, y_train, x_val, y_val, x_test, y_test = ut.preprocess_data()
    print("Preload finished.")

    if training_sys == 'svm':       # SVM accuracy:  0.9803030303030303
        check_spam_svm(x_train, y_train, x_val, y_val, x_test, y_test)
    elif training_sys == 'nn':      # Neural network accuracy:  0.9712121212121212
        check_spam_neural_network(x_train, y_train, x_val, y_val, x_test, y_test)
    elif training_sys == 'log':     # Logistic regression accuracy:  0.9742424242424242
        check_spam_reg_log(x_train, y_train, x_val, y_val, x_test, y_test)

if __name__ == '__main__':
    main('nn')
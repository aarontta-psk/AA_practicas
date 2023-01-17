import logistic_reg as lgr
import svm as svm
import nn as nn

import utils as ut

import time
import numpy as np
import sklearn.metrics as skme

def obtain_reg_log(x_train, y_train, x_val, y_val):
    # initial values
    values = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500])
    w_in, b_in = np.zeros(x_train.shape[1]), 1
    num_iters = 2000
    alpha = 0.01

    # check for best parameters
    print("---------Logistic regression validation begin---------")
    iteration = 0
    accuracies, lambda_list = [], []
    for lambda_value in values:
        print("Logistic regression validation " + str(round(iteration/len(values) * 100, 2)) + "%% out of 100%% done")

        new_w, new_b, _ = lgr.gradient_descent(x_train, y_train, w_in, b_in, lgr.compute_cost_reg, lgr.compute_gradient_reg, alpha, num_iters, lambda_value)

        predict = lgr.predict(x_val, new_w, new_b)
        accuracy = np.sum(y_val == predict) / y_val.shape[0]

        accuracies.append(accuracy)
        lambda_list.append((lambda_value, new_w, new_b))
        iteration += 1

    best_accuracy = np.argmax(accuracies)
    lambda_, w, b = lambda_list[best_accuracy]

    print("---------Logistic regression validation finish---------")
    print("\nLogistic regression best parameters: Lambda --> " + str(lambda_) + "\n")
    
    return w, b

def obtain_neural_network(x_train, y_train, x_val, y_val):
    # initial values
    # values = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500])
    values = [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 50]

    output_labels = 2               # spam or not spam
    input_labels = len(x_train[0])  # nº words email
    hidden_labels = 25              # because I can

    theta1, theta2 = nn.random_thetas(input_labels, hidden_labels, output_labels)
    y_onehot = nn.onehot(y_train, output_labels)
    num_iters = 50

    # check for best parameters
    print("---------Neural network validation begin---------")
    iteration = 0
    accuracies, nn_list = [], []
    for lambda_value in values:
        print("Neural network validation " + str(round(iteration/len(values) * 100, 2)) + "%% out of 100%% done")
         
        theta_1, theta_2 = nn.minimize(x_train, y_onehot, theta1, theta2, num_iters, lambda_value)
        
        predict = nn.predict(theta_1, theta_2, x_val)
        accuracy = np.sum(y_val == predict) / y_val.shape[0]

        accuracies.append(accuracy)
        nn_list.append((lambda_value, theta_1, theta_2))    
        iteration += 1

    best_accuracy = np.argmax(accuracies)
    lambda_, theta1, theta2 = nn_list[best_accuracy]

    print("---------Neural network validation finish---------")
    print("\nNeural network best parameters: Lambda --> " + str(lambda_) + "\n")

    return theta1, theta2

def obtain_svm(x_train, y_train, x_val, y_val):
    # initial values
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
    # check for best parameters
    print("---------SVM validation begin---------")
    iteration = 0
    accuracies, svm_list = [], []
    for c_value in values:
        for sigma_value in values:
            print("SVM validation " + str(round(iteration/(len(values)**2) * 100, 2)) + "%% out of 100%% done")
            new_svm, _, accuracy = svm.svm(x_train, y_train, x_val, y_val, 'rbf', c_value, sigma_value)
            
            accuracies.append(accuracy)
            svm_list.append((c_value, sigma_value, new_svm))
            iteration += 1

    best_accuracy = np.argmax(accuracies)
    c, sigma, svm_ = svm_list[best_accuracy]
    
    print("---------SVM validation finish---------")
    print("\nSVM best parameters: C --> " + str(c) + "; Sigma --> " + str(sigma) + "\n")

    return svm_

def check_spam_reg_log(x_train, y_train, x_val, y_val, x_test, y_test):
    tic = time.perf_counter()
    w, b = obtain_reg_log(x_train, y_train, x_val, y_val)
    predict = lgr.predict(x_test, w, b)
    accuracy = np.sum(y_test == predict) / y_test.shape[0] * 100
    toc = time.perf_counter()
    process_time = toc - tic

    print("Logistic regression accuracy: " + str(round(accuracy, 2)) + "%%")
    print("Logistic regression process time: " + str(round(process_time, 2)) + " seconds\n")
    return accuracy, process_time
    
def check_spam_neural_network(x_train, y_train, x_val, y_val, x_test, y_test):
    tic = time.perf_counter()
    theta1, theta2 = obtain_neural_network(x_train, y_train, x_val, y_val)
    p = nn.predict(theta1, theta2, x_test)
    accuracy = np.sum(y_test == p) / y_test.shape[0] * 100
    toc = time.perf_counter()
    process_time = toc - tic
    
    print("Neural network accuracy: " + str(round(accuracy, 2)) + "%%")
    print("Neural network process time: " + str(round(process_time, 2)) + " seconds\n")
    return accuracy, process_time

def check_spam_svm(x_train, y_train, x_val, y_val, x_test, y_test):
    tic = time.perf_counter()
    svm = obtain_svm(x_train, y_train, x_val, y_val)
    accuracy = skme.accuracy_score(y_test, svm.predict(x_test)) * 100
    toc = time.perf_counter()
    process_time = toc - tic

    print("SVM accuracy: " + str(round(accuracy, 2)) + "%%")
    print("SVM process time: " + str(round(process_time, 2)) + " seconds\n")
    return accuracy, process_time

def main(training_sys):
    x_train, y_train, x_val, y_val, x_test, y_test = ut.preprocess_data()
    print("~~~~~~Preload finished~~~~~~\n")

    if training_sys == 'log':       # Logistic regression accuracy: 96.52
        check_spam_reg_log(x_train, y_train, x_val, y_val, x_test, y_test)
    elif training_sys == 'nn':      # Neural network accuracy: 97.12
        check_spam_neural_network(x_train, y_train, x_val, y_val, x_test, y_test)
    elif training_sys == 'svm':      # SVM accuracy: 98.03
        check_spam_svm(x_train, y_train, x_val, y_val, x_test, y_test)
    elif training_sys == 'all':
        # check every training system
        log_acc, log_time = check_spam_reg_log(x_train, y_train, x_val, y_val, x_test, y_test)
        nn_acc, nn_time = check_spam_neural_network(x_train, y_train, x_val, y_val, x_test, y_test)
        svm_acc, svm_time = check_spam_svm(x_train, y_train, x_val, y_val, x_test, y_test)

        # plot results obtained
        ut.plot_results(log_acc, log_time, nn_acc, nn_time, svm_acc, svm_time)

if __name__ == '__main__':
    main('all')
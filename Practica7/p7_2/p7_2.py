import utils as ut
import numpy as np

import sklearn.model_selection as skm

import sklearn.svm as skl_svm
import sklearn.metrics as skme

import logistic_reg as lgr

# import sys

def transcribe_emails(emails, default_value=0):
    vocab_list = ut.getVocabDict()
    transcribed_emails = np.zeros((len(emails), len(vocab_list)))

    for email in range(len(emails)):
        # ignore non utf-8 characters
        email_contents = open(emails[email], 'r', encoding='utf-8', errors='ignore').read()
        curr_email = ut.email2TokenList(email_contents)
        # sys.exit("Error message")

        for word in range(len(curr_email)):
            if curr_email[word] in vocab_list:
                word_id = vocab_list[curr_email[word]]  - 1
                transcribed_emails[email, word_id] = 1

    return transcribed_emails, np.full(transcribed_emails.shape[0], default_value)

def preprocess():
    easy_ham_x, easy_ham_y = transcribe_emails(ut.obtainAllDirectoryFiles('./data_spam/easy_ham/'))
    hard_ham_x, hard_ham_y = transcribe_emails(ut.obtainAllDirectoryFiles('./data_spam/hard_ham/'))
    spam_x, spam_y =  transcribe_emails(ut.obtainAllDirectoryFiles('./data_spam/spam/'), 1)

    eh_x_train, eh_x_temp, eh_y_train, eh_y_temp = skm.train_test_split(easy_ham_x, easy_ham_y, test_size=0.4, random_state=1)
    eh_x_test, eh_x_val, eh_y_test, eh_y_val = skm.train_test_split(eh_x_temp, eh_y_temp, test_size=0.5, random_state=1)

    hh_x_train, hh_x_temp, hh_y_train, hh_y_temp = skm.train_test_split(hard_ham_x, hard_ham_y, test_size=0.4, random_state=1)
    hh_x_test, hh_x_val, hh_y_test, hh_y_val = skm.train_test_split(hh_x_temp, hh_y_temp, test_size=0.5, random_state=1)

    s_x_train, s_x_temp, s_y_train, s_y_temp = skm.train_test_split(spam_x, spam_y, test_size=0.4, random_state=1)
    s_x_test, s_x_val, s_y_test, s_y_val = skm.train_test_split(s_x_temp, s_y_temp, test_size=0.5, random_state=1)

    x_train, y_train = np.vstack((eh_x_train, hh_x_train, s_x_train)), np.hstack((eh_y_train, hh_y_train, s_y_train))
    x_test, y_test = np.vstack((eh_x_test, hh_x_test, s_x_test)), np.hstack((eh_y_test, hh_y_test, s_y_test))
    x_val, y_val = np.vstack((eh_x_val, hh_x_val, s_x_val)), np.hstack((eh_y_val, hh_y_val, s_y_val))

    return x_train, y_train, x_val, y_val, x_test, y_test

def apply_svm(x_train, y_train, x_val, y_val, c, sigma):
    gamma_value = 1 / (2 * sigma**2)
    svm = skl_svm.SVC(kernel='rbf', C=c, gamma=gamma_value)
    svm.fit(x_train, y_train.ravel())
    predict = svm.predict(x_val)

    return skme.accuracy_score(y_val, predict), svm

def obtain_svm(x_train, y_train, x_val, y_val):
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    svm = skl_svm.SVC()
    best_acc = -1 
    for c in range(len(values)):
        for sigma in range(len(values)):
            accuracy, new_svm = apply_svm(x_train, y_train, x_val, y_val, values[c], values[sigma])
            
            if (best_acc < accuracy):
                best_acc = accuracy
                svm = new_svm
                
    return svm

def obtain_reg_log(x_train, y_train, x_val, y_val):
    w_in = np.zeros(x_train.shape[1])
    b_in = 1

    num_iters = 1000
    alpha = 0.01
    lambdas = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500])
    lambda_ = 0
    w = b = 0
    best_acc = -1 
    for temp_lambda in range(len(lambdas)):
        new_w, new_b, _ = lgr.gradient_descent(x_train, y_train, w_in, b_in, lgr.compute_cost_reg, lgr.compute_gradient_reg, alpha, num_iters, lambdas[temp_lambda])

        predict = lgr.predict(x_val, new_w, new_b)
        accuracy = np.sum(y_val == predict) / y_val.shape[0]

        if(best_acc < accuracy):
            best_acc = accuracy
            w, b, lambda_ = new_w, new_b, lambdas[temp_lambda]

    num_iters = 10000
    w, b, _ = lgr.gradient_descent(x_train, y_train, w_in, b_in, lgr.compute_cost_reg, lgr.compute_gradient_reg, alpha, num_iters, lambda_)

    return w, b

def check_spam_svm(x_train, y_train, x_val, y_val, x_test, y_test):
    svm = obtain_svm(x_train, y_train, x_val, y_val)
    accuracy = skme.accuracy_score(y_test, svm.predict(x_test))
    print("SVM accuracy: ", accuracy)

def check_spam_reg_log(x_train, y_train, x_val, y_val, x_test, y_test):
    w, b = obtain_reg_log(x_train, y_train, x_val, y_val)
    predict = lgr.predict(x_test, w, b)
    accuracy = np.sum(y_test == predict) / y_test.shape[0]
    print("Logistic regression accuracy: ", accuracy)

def main(training_sys='svm'):
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess()
    print("Preload finished.")

    if training_sys == 'svm':
        check_spam_svm(x_train, y_train, x_val, y_val, x_test, y_test)
    elif training_sys == 'nn':
        return 1
    elif training_sys == 'log':
        check_spam_reg_log(x_train, y_train, x_val, y_val, x_test, y_test)

if __name__ == '__main__':
    main('log')
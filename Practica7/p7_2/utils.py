import re
import nltk
import nltk.stem.porter

import os 
import numpy as np
import sklearn.model_selection as skm
# import sys

def preProcess(email):
    hdrstart = email.find("\n\n")
    if hdrstart != -1:
        email = email[hdrstart:]

    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email)
    # Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    # The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email)
    return email

def email2TokenList(raw_email):
    """
    Function that takes in a raw email, preprocesses it, tokenizes it,
    stems each word, and returns a list of tokens in the e-mail
    """
    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess(raw_email)

    # Split the e-mail into individual words (tokens)
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # Loop over each token and use a stemmer to shorten it
    tokenlist = []
    for token in tokens:

        token = re.sub('[^a-zA-Z0-9]', '', token)
        stemmed = stemmer.stem(token)
        # Throw out empty tokens
        if not len(token):
            continue
        # Store a list of all unique stemmed words
        tokenlist.append(stemmed)

    return tokenlist

def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary.
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open("vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    predict : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """

    m = X.shape[0]
    a1 = np.column_stack([np.ones((m, 1)), X])
    z2 = a1 @ theta1.T

    a2 = sigmoid(z2)
    a2 = np.column_stack([np.ones((m, 1)), a2])
    z3 = a2 @ theta2.T

    a3 = sigmoid(z3)
    predict = np.argmax(a3, axis=1)

    return predict

def obtainAllDirectoryFiles(dir):
    return [os.path.join(dir, dir_file) for dir_file in os.listdir(dir) if os.path.isfile(os.path.join(dir, dir_file))]

def transcribe_emails(emails, default_value=0):
    vocab_list = getVocabDict()
    transcribed_emails = np.zeros((len(emails), len(vocab_list)))

    for email in range(len(emails)):
        # ignore non utf-8 characters
        email_contents = open(emails[email], 'r', encoding='utf-8', errors='ignore').read()
        curr_email = email2TokenList(email_contents)
        # sys.exit("Error message")

        for word in curr_email:
            if word in vocab_list:
                word_id = vocab_list[word] - 1
                transcribed_emails[email, word_id] = 1

    return transcribed_emails, np.full(transcribed_emails.shape[0], default_value)

def preprocess_data():
    # adapt emails to what we need for our training (words -> list(0,1))
    easy_ham_x, easy_ham_y = transcribe_emails(obtainAllDirectoryFiles('./data_spam/easy_ham/'))
    hard_ham_x, hard_ham_y = transcribe_emails(obtainAllDirectoryFiles('./data_spam/hard_ham/'))
    spam_x, spam_y = transcribe_emails(obtainAllDirectoryFiles('./data_spam/spam/'), 1)

    # 60% train, 20% validation, 20% test for each type
    eh_x_train, eh_x_temp, eh_y_train, eh_y_temp = skm.train_test_split(easy_ham_x, easy_ham_y, test_size=0.4, random_state=1)
    eh_x_test, eh_x_val, eh_y_test, eh_y_val = skm.train_test_split(eh_x_temp, eh_y_temp, test_size=0.5, random_state=1)

    hh_x_train, hh_x_temp, hh_y_train, hh_y_temp = skm.train_test_split(hard_ham_x, hard_ham_y, test_size=0.4, random_state=1)
    hh_x_test, hh_x_val, hh_y_test, hh_y_val = skm.train_test_split(hh_x_temp, hh_y_temp, test_size=0.5, random_state=1)

    s_x_train, s_x_temp, s_y_train, s_y_temp = skm.train_test_split(spam_x, spam_y, test_size=0.4, random_state=1)
    s_x_test, s_x_val, s_y_test, s_y_val = skm.train_test_split(s_x_temp, s_y_temp, test_size=0.5, random_state=1)

    # we mix the three data sets for each step of the process
    x_train, y_train = np.vstack((eh_x_train, hh_x_train, s_x_train)), np.hstack((eh_y_train, hh_y_train, s_y_train))
    x_test, y_test = np.vstack((eh_x_test, hh_x_test, s_x_test)), np.hstack((eh_y_test, hh_y_test, s_y_test))
    x_val, y_val = np.vstack((eh_x_val, hh_x_val, s_x_val)), np.hstack((eh_y_val, hh_y_val, s_y_val))

    return x_train, y_train, x_val, y_val, x_test, y_test
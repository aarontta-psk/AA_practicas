from copy import deepcopy
import numpy as np
import logistic_reg as lgr

#########################################################################
# one-vs-all
#
def oneVsAll(X, y, n_labels, lambda_):
    """
     Trains n_labels logistic regression classifiers and returns
     each of these classifiers in a matrix all_theta, where the i-th
     row of all_theta corresponds to the classifier for label i.

     Parameters
     ----------
     X : array_like
         The input dataset of shape (m x n). m is the number of
         data points, and n 0is the number of features. 

     y : array_like
         The data labels. A vector of shape (m, ).

     n_labels : int
         Number of possible labels.

     lambda_ : float
         The logistic regularization parameter.

     Returns
     -------
     all_theta : array_like
         The trained parameters for logistic regression for each class.
         This is a matrix of shape (K x n+1) where K is number of classes
         (ie. `n_labels`) and n is number of features without the bias.
     """

    m = X.shape[0]
    n = X.shape[1]

    w_in = np.zeros(n)
    b_in = 0
    alpha = 1
    n_iters = 1500

    all_theta = np.zeros([n_labels, n + 1])

    for i in range(n_labels):
        classif = (y == i)
        w, b, _ = lgr.gradient_descent(X, classif, w_in, b_in, lgr.compute_cost_reg, lgr.compute_gradient_reg,  alpha, n_iters, lambda_)
        all_theta[i] = np.append(b, w)
    
    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """

    m = X.shape[0]

    p = np.zeros(m)

    for i in range(m):
        p[i] = np.argmax(lgr.sigmoid(np.append(1, X[i]) @ all_theta.T))

    return p


#########################################################################
# NN
#
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
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """

    m = X.shape[0]
    a1 = np.column_stack([np.ones((m, 1)), X])
    z2 = a1 @ theta1.T

    a2 = lgr.sigmoid(z2)
    a2 = np.column_stack([np.ones((m, 1)), a2])
    z3 = a2 @ theta2.T

    a3 = lgr.sigmoid(z3)
    p = np.argmax(a3, axis=1)

    return p

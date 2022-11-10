import numpy as np
import logistic_reg as lgr

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """

    m = X.shape[0]

    a1 = np.column_stack([np.ones((m, 1)), X])
    z2 = a1 @ theta1.T

    a2 = lgr.sigmoid(z2)
    a2 = np.column_stack([np.ones((m, 1)), a2])
    z3 = a2 @ theta2.T

    h0 = a3 = lgr.sigmoid(z3)

    t1 = (y * np.log(h0))
    t2 = ((1 - y) * np.log(1 - h0))

    cost = t1 + t2
    reg_factor = np.sum(pow(theta1[:, 1:], 2)) + np.sum(pow(theta2[:, 1:], 2))
    J = (-1 / m) * np.sum(cost)
    J += (lambda_ / (2 * m)) * reg_factor

    return J



def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """


    return (J, grad1, grad2)


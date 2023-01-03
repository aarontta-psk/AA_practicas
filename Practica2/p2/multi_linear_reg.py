import numpy as np
import copy

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """

    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter

    Returns
      cost (scalar)    : cost
    """

    m = X.shape[0]

    # unoptimized (iterative)
    # cost = 0
    # for i in range(m):
    #     f_wb = np.dot(w, X[i]) + b
    #     cost_i = (f_wb - y[i]) ** 2
    #     cost += cost_i

    # optimized (vectorized)
    f_wb = X @ w + b
    cost = np.sum((f_wb - y) ** 2)

    cost /= 2 * m

    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model

    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """

    m = X.shape[0]

    # unoptimized (iterative)
    # dj_dw = 0
    # dj_db = 0
    # for i in range(m):
    #     f_wb = np.dot(w, X[i]) + b
    #     dj_dw_i = np.dot((f_wb - y[i]), X[i])
    #     dj_db_i = (f_wb - y[i])

    #     dj_dw += dj_dw_i
    #     dj_db += dj_db_i

    # optimized (vectorized)
    f_wb = X @ w + b
    dj_dw = X.T @ (f_wb - y)
    dj_db = np.sum(f_wb - y)

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent

    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model
          after running gradient descent
      b : (scalar)                Updated value of parameter of the model 
          after running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

    J_history = []
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)

    for _ in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = cost_function(X, y, w, b)
        J_history.append(cost) 

    return w, b, J_history
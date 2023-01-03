import copy
import numpy as np

#########################################################################
# Cost function
#
def compute_cost(X, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        cost (float): The cost of using w, b as the parameters for linear regression
               to fit the data points in x and y
    """

    # the shape attribute returns the dimensions of the array [x.shape = (n,m)]
    m = X.shape[0]

    # unoptimized (iterative)
    # cost = 0
    # for i in range(m):
    #     f_wb = w * x[i] + b
    #     cost_i = (f_wb - y[i]) ** 2
    #     cost += cost_i

    # optimized (vectorized)
    f_wb = w * X + b
    cost = np.sum((f_wb - y) ** 2)

    cost /= 2 * m

    return cost

#########################################################################
# Gradient function
#
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model
      
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """

    # the shape attribute returns the dimensions of the array [x.shape = (n,m)]
    m = X.shape[0]

    # unoptimized (iterative)
    # dj_dw = 0
    # dj_db = 0
    # for i in range(m):
    #     f_wb = w * x[i] + b
    #     dj_dw_i = (f_wb - y[i]) * x[i]
    #     dj_db_i = (f_wb - y[i])

    #     dj_dw += dj_dw_i
    #     dj_db += dj_db_i

    # optimized (vectorized)
    f_wb = w * X + b
    dj_dw = np.sum((f_wb - y) * X)
    dj_db = np.sum((f_wb - y))

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

#########################################################################
# Gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent

    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
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
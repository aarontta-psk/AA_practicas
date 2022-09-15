from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def integra_mc(funct, a, b, num_points=10000):
    # empty function array 
    function_dots = np.empty(shape=1000)

    # function brute force
    brute_force_values = 1000
    step = (b - a) / brute_force_values
    for i in range(0, brute_force_values):
        function_dots[i] = funct(a + step * i)

    # obtain max value
    max_value =  np.max(function_dots)

    # montecarlo integrate
    return iterative_mode(funct, a, b, max_value, num_points)
    return vector_mode(funct, a, b, max_value, num_points)

def iterative_mode(funct, a, b, max_value, num_points):
    montecarlo_dots = np.empty(shape=(num_points, 2))
    count = 0

    for i in range(0, num_points):
        montecarlo_dots[i, 0] = uniform(a, b)
        montecarlo_dots[i, 1] = uniform(0, max_value)
        
        if(funct(montecarlo_dots[i, 0]) >= montecarlo_dots[i, 1]): 
            count += 1
        
    return (count / num_points) * ((b - a) * max_value)

def vector_mode(funct, a, b, max_value, num_points):
    # result values from the given function 
    function_dots_y = funct(np.random.uniform(a, b, num_points))
    # random values with function max cap
    montecarlo_dots_y = np.random.uniform(0, max_value, num_points)
    # number of dots below function
    count = sum(montecarlo_dots_y < function_dots_y)

    return (count / num_points) * ((b - a) * max_value)

def squared(x):
    return x * x

def main():
    print(integra_mc(squared, 50, 3000))
    print(integrate.quad(squared, 50, 3000))

main()
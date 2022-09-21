import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def obtain_time(funct, a, b, v_mode=False, num_points=10000):
    tic = time.process_time()
    integrate_mc(funct, a, b, v_mode, num_points)
    toc = time.process_time()
    return 1000 * (toc - tic)

def time_test(funct, a, b):
    test_sizes = np.linspace(100, 1000000, 20)
    it_times = []
    vec_times = []

    for size in test_sizes:
        it_times += [obtain_time(funct, a, b, False, int(size))]
        vec_times += [obtain_time(funct, a, b, True, int(size))]

    plt.figure()
    plt.scatter(test_sizes, it_times, c='red', label='iterative')
    plt.scatter(test_sizes, vec_times, c='blue', label='vectorized')
    plt.legend()
    plt.savefig('graph.pdf')

def iterative_mode(funct, a, b, max_value, num_points):
    montecarlo_dots = np.empty(shape=(num_points, 2))
    count = 0

    for i in range(0, num_points):
        montecarlo_dots[i, 0] = np.random.uniform(a, b)
        montecarlo_dots[i, 1] = np.random.uniform(0, max_value)
        
        if(funct(montecarlo_dots[i, 0]) >= montecarlo_dots[i, 1]): 
            count += 1
        
    return (count / num_points) * ((b - a) * max_value)

def vector_mode(funct, a, b, max_value, num_points):
    # result values from the given function 
    function_dots_y = funct(np.random.uniform(a, b, num_points))
    # random values with function max cap
    montecarlo_dots_y = np.random.uniform(0, max_value, num_points)
    # number of dots below function
    count = np.sum(montecarlo_dots_y < function_dots_y)

    return (count / num_points) * ((b - a) * max_value)

def integrate_mc(funct, a, b, v_mode=False, num_points=10000):
    # empty function array 
    function_dots = np.empty(shape=1000)

    # function brute force
    brute_force_values = 1000
    step = (b - a) / brute_force_values
    for i in range(0, brute_force_values):
        function_dots[i] = funct(a + step * i)

    # obtain max value
    max_value = np.max(function_dots)

    # montecarlo integrate result
    if v_mode == False: 
        return iterative_mode(funct, a, b, max_value, num_points)
    return vector_mode(funct, a, b, max_value, num_points)

def squared(x):
    return x * x

def cube(x):
    return x ** 3

def main():
    time_test(squared, 50, 3000)
    # print(integrate_mc(cube, 50, 3000))
    # print(integrate.quad(cube, 50, 3000))

if __name__ == '__main__':
    main()
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# test functions
def squared(x):
    return x * x

def cube(x):
    return x ** 3

# auxiliary function to obtain execution time
def obtain_time(funct, a, b, num_points=10000, vec_mode=False):
    tic = time.process_time()
    integrate_mc(funct, a, b, num_points, vec_mode)
    toc = time.process_time()
    return 1000 * (toc - tic)

# montecarlo integration using iteration
def iterative_mode(funct, a, b, num_points, max_value):
    # first row the 'x' axis, second row the 'y' axis
    montecarlo_dots = np.empty(shape=(num_points, 2))
    count = 0

    for i in range(0, num_points):
        # we obtain the random point in both axis
        montecarlo_dots[i, 0] = np.random.uniform(a, b)
        montecarlo_dots[i, 1] = np.random.uniform(0, max_value)
        
        # if lower than the function 'y' axis value, add to the counter
        if(funct(montecarlo_dots[i, 0]) >= montecarlo_dots[i, 1]): 
            count += 1
    
    # return the integrate value
    return (count / num_points) * ((b - a) * max_value)

# montecarlo integration using vectorization
def vector_mode(funct, a, b, num_points, max_value):
    # result values from the given function 
    function_dots_y = funct(np.random.uniform(a, b, num_points))
    # random values with function max cap
    montecarlo_dots_y = np.random.uniform(0, max_value, num_points)
    # number of dots below function
    count = np.sum(montecarlo_dots_y < function_dots_y)

    # return the integrate value
    return (count / num_points) * ((b - a) * max_value)

# function to test speed difference between both types of integration
def time_test(funct, a, b):
    # different sizes to test speed
    test_sizes = np.linspace(100, 1000000, 20)
    it_times = []
    vec_times = []

    # obtain speed for each size used
    for size in test_sizes:
        it_times += [obtain_time(funct, a, b, int(size), False)]
        vec_times += [obtain_time(funct, a, b, int(size), True)]

    # print results
    plt.figure()
    plt.scatter(test_sizes, it_times, c='red', label='iterative')
    plt.scatter(test_sizes, vec_times, c='blue', label='vectorized')
    plt.legend()
    plt.savefig('graph.pdf')

def integrate_mc(funct, a, b, num_points=10000, vec_mode=False):
    # empty function array 
    function_dots = np.empty(shape=1000)

    # function max value brute force
    brute_force_values = 1000
    step = (b - a) / brute_force_values
    for i in range(0, brute_force_values):
        function_dots[i] = funct(a + step * i)

    # obtain max value
    max_value = np.max(function_dots)

    # montecarlo integrate result
    if vec_mode == False: 
        return iterative_mode(funct, a, b, num_points, max_value)
    return vector_mode(funct, a, b, num_points, max_value)

# main function
def main(check_time=True):
    if check_time == True:
        time_test(squared, 50, 3000)
    else:
        print(integrate_mc(cube, 50, 3000, 10000, True))
        print(integrate.quad(cube, 50, 3000))

if __name__ == '__main__':
    main()
# Author: Arnar Gylfi Haraldsson   
# Date: 13/09/2023  
# Project: 
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    I_k = np.identity(k)
    X = np.random.multivariate_normal(mean, (var**2)*I_k,n)
    return X
np.random.seed(1234)
X = gen_data(300,3,[0,1,-1],np.sqrt(3))

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update'''
    updated_mu = mu+(x-mu)/n
    return updated_mu
mean = np.mean(X, 0)
new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)



def _plot_sequence_estimate():
    data = gen_data(100,3,[0,0,0],4) # Set this as the data
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1],data[i],i+1)
        estimates.append(new_estimate)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')

    plt.legend(loc='upper center')
    plt.show()
# _plot_sequence_estimate()

def _square_error(y, y_hat):
    sqr_e = (y - y_hat)**2
    return sqr_e

def _plot_mean_square_error():
    np.random.seed(1234)
    actual_mean = [0, 0, 0]
    data = gen_data(100,3,[0,0,0],4)
    estimates = [np.array([0, 0, 0])]
    avg_errors = []

    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1], data[i], i + 1)
        estimates.append(new_estimate)
        
        # Calculate squared error and average across dimensions
        sq_error = _square_error(new_estimate, actual_mean)
        avg_error = np.mean(sq_error)
        avg_errors.append(avg_error)

    # Plot the average squared error
    plt.plot(avg_errors, label='Average Squared Error')
    plt.legend()
    plt.title("Average Squared Error vs. Number of Data Points")
    plt.xlabel("Number of Data Points")
    plt.ylabel("Average Squared Error")
    plt.show()








# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass

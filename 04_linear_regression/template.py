# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal



def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    N, M = features.shape[0], mu.shape[0]
    fi = np.zeros((N, M))

    for i in range(M):
        mvn = multivariate_normal(mean=mu[i], cov=np.eye(mu[i].shape[0]) * sigma)

        fi[:, i] = mvn.pdf(features)

    return fi

X, t = load_regression_iris()
N, D = X.shape
M, sigma = 10, 10
mu = np.zeros((M, D))
for i in range(D):
    mmin = np.min(X[i, :])
    mmax = np.max(X[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)
fi = mvn_basis(X, mu, sigma)


def _plot_mvn():
    plt.figure(figsize=(10, 6))
    for i in range(M):
        plt.plot(X, fi[:, i], label=f'Basis {i+1}')
    plt.legend()
    plt.title("Output of Gaussian Basis Functions")
    plt.xlabel("Features")
    plt.ylabel("Basis Function Output")
    plt.grid(True)
    plt.savefig("plot_1_2_1.png")
    plt.show()
# _plot_mvn()


def max_likelihood_linreg(fi, targets, lamda):
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * fi: [NxM] is the array of basis function vectors
    * targets: [Nx1] is the target value for each input in fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    N, M = fi.shape
    I = np.identity(M)
    wml = np.linalg.inv(fi.T @ fi + lamda * I) @ fi.T @ targets
    
    return wml

lamda = 0.001
wml = max_likelihood_linreg(fi, t, lamda)


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    N, M = features.shape
    fi = mvn_basis(features, mu, sigma)  

    prediction = fi @ w

    return prediction

wml = max_likelihood_linreg(fi, t, lamda) # as before
prediction = linear_model(X, mu, sigma, wml)
print(prediction)

#SECTION 1.5

plt.figure(figsize=(10, 6))
plt.plot(X, t, label='Real Values', color='blue')
plt.plot(X, prediction, label='Predictions', color='green')
plt.legend()
plt.xlabel("Features")
plt.ylabel("Values")
plt.grid(True)
plt.show()

error = abs(prediction-t)/t
# print(error)

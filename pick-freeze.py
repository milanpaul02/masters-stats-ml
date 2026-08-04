#Made with AI assistance

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

#we first define the underlying model - the vertical displacement of a 2-dimensional projectile that travels a distance D, launched at velocity v at angle theta under a wind velocity w and acceleration due to gravity g

#The input X has five components: X[0] = v, X[1] = theta, X[2] = g, X[3] = w, X[4] = D. The output is the vertical displacement of the projectile.

def G(X):
  v = X[0]
  theta = X[1]
  g = X[2]
  w = X[3]
  D = X[4]

  #compute the vertical displacement of the projectile
  Y = v*np.sin(theta)*D/(v*np.cos(theta) + w) - 0.5*g*D**2/(v*np.cos(theta) + w)**2

  return Y

#Generate a sample 1000 inputs X from a multivariate normal distribution with mean [1200, 0, 9.81, 0, 250] and covariance matrix [[1, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 0.1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

#implementation of the pick-freeze scheme to compute Sobol' indices

def pick_freeze(G, A, X, X_prime):
  n = X.shape[0]
  d = X.shape[1]

  #compute the output of the model for the two input matrices

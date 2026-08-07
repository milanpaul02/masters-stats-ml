#Made with AI assistance

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

#we first define the underlying model - the vertical displacement of a 2 dimensional projectile that travels a distance D launched at velocity v at angle theta under a wind velocity w and acceleration due to gravity g

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

#implementation of the pick-freeze scheme to compute Sobol' indices

def pick_freeze(G, j, X, X_prime):
  n = X.shape[0]
  d = X.shape[1]

  #j is the index of the variable for which we want to compute the Sobol' index
  
  #define X_j as the matrix formed by the replacing the columns of X_prime with the columns of X for the j-th variable
  X_j = X_prime.copy()
  X_j[:, j] = X[:, j]

  #Let X^i be the i-th row of X, X_j^i be the i-th row of X_j, and X_prime^i be the i-th row of X_prime.
  #Then the estimator of the Sobol' index S_j is given by the average over i of G(X^i) * (G(X_j^i) - G(X_prime^i)) divided by the variance of G(X).

  S_j = 0
  for i in range(n): S_j += G(X[i, :]) * (G(X_j[i, :]) - G(X_prime[i, :]))
  S_j /= n * np.var([G(x) for x in X], ddof=1)
  return S_j

#Generate a sample 50000 inputs X from a multivariate normal distribution with mean [900, 0, 9.81, 0, 250] and covariance matrix [[1, 0, 0, 0, 0], [0, np.pi/8, 0, 0, 0], [0, 0, 0.01, 0, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 50]]
X = scs.multivariate_normal.rvs(mean=[900, 0, 9.81, 0, 250], cov=[[1, 0, 0, 0, 0], [0, np.pi/8, 0, 0, 0], [0, 0, 0.01, 0, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 50]], size=50000)
X_prime = scs.multivariate_normal.rvs(mean=[900, 0, 9.81, 0, 250], cov=[[1, 0, 0, 0, 0], [0, np.pi/8, 0, 0, 0], [0, 0, 0.01, 0, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 50]], size=50000)

for i in range(X.shape[0]):
  #Ensure that the angle theta is between -pi/8 and pi/8
  if X[i, 1] < -np.pi/8:
    X[i, 1] = -np.pi/8
  elif X[i, 1] > np.pi/8:
    X[i, 1] = np.pi/8

  if X_prime[i, 1] < -np.pi/8:
    X_prime[i, 1] = -np.pi/8
  elif X_prime[i, 1] > np.pi/8:
    X_prime[i, 1] = np.pi/8

  #ensure that the distance D is between 200 and 300
  if X[i, 4] < 200:
    X[i, 4] = 200
  elif X[i, 4] > 300:
    X[i, 4] = 300

  if X_prime[i, 4] < 200:
    X_prime[i, 4] = 200
  elif X_prime[i, 4] > 300:
    X_prime[i, 4] = 300

  #ensure that the wind velocity w is between -10 and 10
  if X[i, 3] < -10:
    X[i, 3] = -10
  elif X[i, 3] > 10:
    X[i, 3] = 10

  if X_prime[i, 3] < -10:
    X_prime[i, 3] = -10
  elif X_prime[i, 3] > 10:
    X_prime[i, 3] = 10

  #ensure that the acceleration due to gravity g is between 9.7 and 9.9
  if X[i, 2] < 9.7:
    X[i, 2] = 9.7
  elif X[i, 2] > 9.9:
    X[i, 2] = 9.9

  if X_prime[i, 2] < 9.7:
    X_prime[i, 2] = 9.7
  elif X_prime[i, 2] > 9.9:
    X_prime[i, 2] = 9.9

  #ensure that the velocity v is between 899 and 901
  if X[i, 0] < 899:
    X[i, 0] = 899
  elif X[i, 0] > 901:
    X[i, 0] = 901

  if X_prime[i, 0] < 899:
    X_prime[i, 0] = 899
  elif X_prime[i, 0] > 901:
    X_prime[i, 0] = 901

for j in range(5):
  #Compute the Sobol' index for the j-th variable using the pick-freeze scheme
  S_j = pick_freeze(G, j, X, X_prime)

  print(f"Sobol' index for variable {j}: {S_j}")

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

## Defining the model that we will use to generate the data

#The model is a gaussian vector (X,Y) with mean (0,0) and covariance matrix [[1,0.3],[0.3,1]]

def generate_data(rho, n_samples):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = scs.multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples)
    return data

##Estimating the parameters of the model using Bayesian inference for different choices of prior distributions

#Generate a dataset of 500 samples with rho=0.3
data = generate_data(rho=0.3, n_samples=20)

#Prior distributions for rho

prior1 = scs.uniform(-1, 2)  # Uniform prior between -1 and 1
prior2 = scs.beta(a=2, b=2, loc=-1, scale=2)  # Beta prior transformed to [-1, 1] with a=2, b=2
prior3 = scs.beta(a=2, b=5, loc=-1, scale=2)  # Beta prior transformed to [-1, 1] with a=2, b=5

#a function to compute the posterior distribution of rho given the data and a prior distribution

def posterior(rho, data, prior):
    n_sample = data.shape[0]
    #Compute the Monte-Carlo estimate of the integral of the product of the likelihood and the prior over the parameter space
    S = 0
    for i in range(1000):
        #Generate a random variable from the prior distribution
        rho_sample = prior.rvs()
        #Compute the product of the likelihood and the prior for the sampled value of rho
        likelihood = np.prod(scs.multivariate_normal.pdf(data, mean=[0, 0], cov=[[1, rho_sample], [rho_sample, 1]]))
        S += likelihood * prior.pdf(rho_sample)
    I = S / 1000
    return prior.pdf(rho) * np.prod(scs.multivariate_normal.pdf(data, mean=[0, 0], cov=[[1, rho], [rho, 1]])) / I

#Plotting the posterior distributions for different choices of prior distributions and for data samples of size 1, 5, 10 and 20

x = np.linspace(-1, 1, 100)
n_samples_list = [1, 5, 10, 20]     
posterior1 = np.vectorize(lambda rho: posterior(rho, data[:n_samples_list[0]], prior1))(x)
posterior2 = np.vectorize(lambda rho: posterior(rho, data[:n_samples_list[1]], prior2))(x)
posterior3 = np.vectorize(lambda rho: posterior(rho, data[:n_samples_list[2]], prior3))(x)
posterior4 = np.vectorize(lambda rho: posterior(rho, data[:n_samples_list[3]], prior1))(x)

#Plotting the posterior distributions
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, posterior1, label='Posterior with Uniform Prior (n=1)')
plt.title('Posterior Distribution with Uniform Prior (n=1)')
plt.xlabel('rho')
plt.ylabel('Posterior Density')
plt.subplot(2, 2, 2)
plt.plot(x, posterior2, label='Posterior with Beta Prior (n=5)')
plt.title('Posterior Distribution with Beta Prior (n=5)')
plt.xlabel('rho')
plt.ylabel('Posterior Density')
plt.subplot(2, 2, 3)
plt.plot(x, posterior3, label='Posterior with Beta Prior (n=10)')
plt.title('Posterior Distribution with Beta Prior (n=10)')
plt.xlabel('rho')
plt.ylabel('Posterior Density')
plt.subplot(2, 2, 4)
plt.plot(x, posterior4, label='Posterior with Uniform Prior (n=20)')
plt.title('Posterior Distribution with Uniform Prior (n=20)')
plt.xlabel('rho')
plt.ylabel('Posterior Density')



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

#an implementation of the CQR method as described in the paper "Conformalized Quantile Regression" by Romano et al. (2019)

def model(n, a0, a1, b):
    X = np.zeros((n, 3))
    y = np.zeros(n)
    for i in range(n):
        X[i, 0] = 2*scs.uniform.rvs() - 1
        X[i, 1] = 2*scs.uniform.rvs() - 1
        X[i, 2] = 1
        epsilon = scs.cauchy.rvs()
        y[i] = a0*X[i, 0] + a1*X[i, 1] + b + epsilon
    return X, y

X_reg, y_reg = model(500, 2, 1, 1)

X_test, y_test = model(200, 2, 1, 1)

#we compute sigmahat2 and betahat - for multivariate linear regression

betahat = (np.linalg.inv(np.transpose(X_reg) @ X_reg) @ np.transpose(X_reg)) @ y_reg

y_reghat = X_reg @ betahat

sigmahat2 = np.sum((y_reg - y_reghat)*(y_reg - y_reghat))/(len(y_reg) - 3)

y_pred = np.zeros(len(y_test))
ci_l = np.zeros(len(y_test))
ci_h = np.zeros(len(y_test))
ci_conf_l = np.zeros(len(y_test))
ci_conf_h = np.zeros(len(y_test))

S = 0

#compute the CQR scores for each element in X_conf and find the 0.975th quantile
for i in range(len(y_test)):
    X_conf, y_conf = model(200, 2, 1, 1)
    score = []
    for j in range(len(y_conf)):
        ytemp = np.transpose(X_conf[j]) @ betahat
        ci_l_temp = ytemp - scs.t.ppf(0.975, df = len(y_reg) - 3)*np.sqrt(sigmahat2*(1 + (np.transpose(X_conf[j]) @ np.linalg.inv(np.transpose(X_reg) @ X_reg)) @ X_conf[j]))
        ci_h_temp = ytemp + scs.t.ppf(0.975, df = len(y_reg) - 3)*np.sqrt(sigmahat2*(1 + (np.transpose(X_conf[j]) @ np.linalg.inv(np.transpose(X_reg) @ X_reg)) @ X_conf[j]))
        scorej = max(ci_l_temp - y_conf[j], y_conf[j] - ci_h_temp)
        score.append(scorej)

    sorted_scores = np.sort(score)
    qhat = scs.mstats.mquantiles(sorted_scores, 0.95*(1 + 1/len(y_conf)))[0]

    #compute the CQR confidence interval for each element in X_test

    y_pred[i] = np.transpose(X_test[i]) @ betahat
    ci_l[i] = y_pred[i] - scs.t.ppf(0.975, df = len(y_reg) - 3)*np.sqrt(sigmahat2*(1 + (np.transpose(X_test[i]) @ np.linalg.inv(np.transpose(X_reg) @ X_reg)) @ X_test[i]))
    ci_h[i] = y_pred[i] + scs.t.ppf(0.975, df = len(y_reg) - 3)*np.sqrt(sigmahat2*(1 + (np.transpose(X_test[i]) @ np.linalg.inv(np.transpose(X_reg) @ X_reg)) @ X_test[i]))
    ci_conf_l[i] = ci_l[i] - qhat
    ci_conf_h[i] = ci_h[i] + qhat

    if ci_conf_l[i] <= y_test[i] <= ci_conf_h[i]: S += 1

#we compute the proper training conditional coverage of the CQR confidence intervals
print("CQR proper training conditional coverage: ", S/len(y_test))

#we plot the first 50 elements of y_test and the corresponding CQR confidence intervals

indices = np.arange(50)
plt.figure(figsize=(10, 6))
plt.plot(indices, y_test[:50], 'ro', label='y_test', markersize=8)
plt.errorbar(indices, y_pred[:50], yerr=[y_pred[:50] - ci_conf_l[:50], ci_conf_h[:50] - y_pred[:50]], fmt='o', color='gray', ecolor='lightgray', elinewidth=3, capsize=0, label='CQR Confidence Interval')
plt.fill_between(indices, ci_conf_l[:50], ci_conf_h[:50], color='lightblue', alpha=0.5, label='CQR Confidence Interval Area')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('CQR Confidence Intervals vs True Values of y_test')
plt.grid()
plt.show()      

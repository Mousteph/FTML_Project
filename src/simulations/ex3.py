import numpy as np
import matplotlib.pyplot as plt

# set random seed
np.random.seed(42)

# functions given on TP1
def generate_output_data(X, theta_star, sigma):
    noise = np.random.normal(0, sigma, size=(n, 1))
    Y = np.matmul(X, theta_star)+noise
    return Y

def OLS_estimator(X, Y):
    covariance_matrix = np.matmul(np.transpose(X), X)
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = np.matmul(inverse_covariance, np.matmul(np.transpose(X), Y))
    return theta_hat

def error(theta_hat, X, Y):
    Y_predictions = np.matmul(X, theta_hat)
    return (np.linalg.norm(Y-Y_predictions))**2

# compute estimation of variance
def var_esti(Y, X):
    theta_hat = OLS_estimator(X, Y)
    return error(theta_hat, X, Y) / (n - d)

# choosen parameters
n_list = range(30, 400, 5)
sigma = 0.2
n = 400
d = 10

# generate X
X = np.random.rand(n, d)

# Bayes predictor
theta_star = np.random.rand(d).reshape(d, 1)

# generate Y using X and theta_star
Y = generate_output_data(X, theta_star, sigma)

bayes_risk = sigma**2

# Compute the estimation depending on the number of samples
risks = []
for n in n_list:
    risks.append(var_esti(Y[:, :n], X[:, :n]))

# plot the result
color = "darkmagenta"
plt.plot(n_list, risks, "o", label=r"$E[\frac{||Y - X\hat{\theta}||^2}{n - d}]$", color=color, markersize=3, alpha=0.6)
plt.xlabel("n")
plt.ylabel("Expected value")
plt.plot(n_list, [bayes_risk]*len(n_list),
         label="Bayes risk: "+r"$\sigma^2$", color="aqua")
plt.title("Experimental estimation of $\sigma^2$")
plt.legend(loc="best")
plt.show()
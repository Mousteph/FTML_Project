import numpy as np
import matplotlib.pyplot as plt

# number of samples
n = 3000

# set random seed
np.random.seed(42)

# generate the law of X
X = np.random.randint(0, 3, n) + np.random.binomial(1, 2/3, n)

# genreate the law of Y depending on X
def generate_Y(x):
    if x == 0:
        return np.random.binomial(1, np.pi / 4)
    elif x == 1:
        return np.random.binomial(1, np.sqrt(2) / 2)
    elif x == 2:
        return np.random.binomial(1, np.exp(1) / 3)
    elif x == 3:
        return np.random.binomial(1, - np.cos(42))
    
Y = np.array(list(map(generate_Y, X)))

# get empirical risk depending on number of samples
empirical_risk = []

size = 0
n_sum = 0
for y in Y:
    size += 1
    n_sum += y
    empirical_risk.append(1 - n_sum / size)
    
# plot the result
plt.scatter(np.arange(n), empirical_risk, s=3, label="Rn(f) empirical risk")
plt.axhline(0.29, color="red", alpha=0.7, label="real risk / generalization error")
plt.legend(loc="best")
plt.xlabel("n")
plt.ylim([0, 1])
plt.title("f: Empirical risk and generalization error\nR(f)=0.29")
plt.show()
# TASK 10
# TASK: Using the Central Limit Theorem (CLT) to show that means from randomly selected samples
# from a uniform distribution tend to a normal distribution.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # library for statistical tests

df = pd.read_csv("uniform_distribution_data.csv")
print(df.head())


def ctg(data):
    means = []
    for _ in range(5000):
        sample = np.random.choice(data["Value"], size=100)
        means.append(np.mean(sample))
    # iterator is not needed here (that's why _)
    # we draw 100 random samples from a specific Value column
    return means


# print(ctg(df))
plt.hist(ctg(df), bins=50)
plt.show()
# with each run the result is a bit different, but the graph itself looks similar

stat, p = stats.shapiro(ctg(df))
print(stat)  # ~1
print(p)  # p > 0.05 THEN hypothesis H0 OK

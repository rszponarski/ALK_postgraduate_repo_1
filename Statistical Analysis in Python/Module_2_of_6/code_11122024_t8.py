# TASK 8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data generation function
def generate_binomial_data(n, p):
    return pd.DataFrame({"Data": np.random.binomial(n, p, size=1000)})


if __name__ == "__main__":  # condition checks if the script is run directly (and not imported as a module)

    # Loading of binomial distribution parameters
    n = int(1000)    # int(input())
    p = float(0.5)  # float(input())

    # Data generation
    data = generate_binomial_data(n,p)

    # Histogram generation
    plt.hist(data['Data'], bins=50, edgecolor='k')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Binomial Distribution Data')
    plt.show() #plt.saveimg()

    # Display of statistics
    print(data.describe())

'''
NOTES:
generate_binomial_data(n, p):
    Function generating data from a binomial distribution.
n:  Number of trials (e.g. number of draws, coin tosses).
p:  Probability of success in each trial (e.g. probability of getting heads in a single coin toss).

*np.random.binomial(n, p, size=1000):
    Generates 1000 samples from a binomial distribution with
    parameters n (number of trials) and p (probability of success in each trial).
    
pd.DataFrame(...): Converts the result to a DataFrame object to make it easier to manage data in pandas.
'''
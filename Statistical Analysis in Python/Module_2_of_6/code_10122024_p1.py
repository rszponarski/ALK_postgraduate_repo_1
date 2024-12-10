# TASK 5- Normal distribution simulation
import numpy as np
import pandas as pd
'''Parameters describing the normal distribution:
n - number of samples,
s - mean,
d - standard deviation.

The code below generates random data based on the normal distribution (Gaussian distribution)
and stores it in a DataFrame object from the pandas library.
'''


def normal_dist(s, d, n):
    return np.random.normal(s, d, n)  # Returns an array (numpy.ndarray)
    # mean, std_dev, no_samples


mean = int(input("Input mean: "))
standard_dev = int(input("Input standard deviation: "))
number_of_samples = int(input("Input number of samples: "))

data = pd.DataFrame({"Data": normal_dist(mean, standard_dev, number_of_samples)})

print(data.describe())

print(type(normal_dist(mean, standard_dev, number_of_samples)))
# print(normal_dist(mean, standard_dev, number_of_samples)) return list of all samples

'''
np.random.normal method comes from the NumPy library;
-used to generate random numbers from a normal distribution (Gaussian distribution)
'''

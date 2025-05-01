# TASK 7
import pandas as pd
from scipy import stats

# Data reading
data = pd.read_csv('distributions.csv')

for column in data.columns:  # Loop through each column in the data set
    stat, p = stats.shapiro(data[column])
    print(f"{column}: statistics={stat}, p-value={p}")

print("Normal distribution: data_1")

'''
* scipy.stats: A module in the SciPy library that contains statistical functions.
It contains hypothesis tests, probability distributions, and other statistical analysis tools.
In this code we use the shapiro function to test for normality.

1. code parses data from a CSV file and performs a normality test (Shapiro-Wilk)
on each column of data to check whether the data in a given column comes from a normal distribution.

2. stats.shapiro(data[column]): The Shapiro-Wilk test function is used to test for normality in the data.
It returns two values:

--- stat:  The Shapiro-Wilk test statistic. This is a value that measures how well the data fits a normal distribution.
The closer to 1, the more the data fits a normal distribution.

--- p:     The p-value. Determines whether the results of the test are statistically significant.
If the p-value is less than a specified threshold (usually 0.05), we reject the null hypothesis,
which is the hypothesis that the data is normally distributed.

If the p-value is less than 0.05, it means that the data is probably not normally distributed.
'''


# TASK 11

import pandas as pd
import matplotlib.pyplot as plt\

# Loading data from csv
df = pd.read_csv('experiment_data.csv')

# Calculation of survival rate
df["Survival rate"] = df["Number of cells surviving"]/df["Initial number of cells"]
# For each sample in the data, calculate the survival rate as the ratio of the number
# of surviving cells to the initial number of cells.

# Calculation of the average survival rate
mean_survival_rate = df["Survival rate"].mean()

# Histogram generating
plt.hist(df['Survival rate'], bins=50, edgecolor='black')
plt.title('Histogram of Cell Survival Rate')
plt.xlabel('Survival rate')
plt.ylabel('Frequency')
plt.show()

print(f"Mean survival rate: {mean_survival_rate:.2f}")
# .2f: Average survival rate rounded to two decimal places.

"""
NOTES:
ad.  plt.hist(df['Survival rate'], bins=50, edgecolor='black')

__plt.hist: Matplotlib function to generate a histogram.
__df['Survival rate']: Data (survival rate) used to generate the histogram.
__bins=50: Number of bins in the histogram. Data is divided into 50 equal bins.
__edgecolor='black': Adds a black border around each bar in the histogram, which improves readability.
"""

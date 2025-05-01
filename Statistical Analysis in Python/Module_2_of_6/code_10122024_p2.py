import numpy as np
'''
TASK 6
Identifying samples that appear to deviate from the typical expression pattern,
--> those that are outside the range of two standard deviations from the mean.
'''

# Loading data from a CSV file into a NumPy table
data = np.loadtxt("genes_data.csv", delimiter=",", skiprows=1)
    # np.loadtxt: Loads data from a text file into a NumPy array.
    # delimiter=",": Sets a comma as the column separator (CSV file).
    # skiprows=1: Skips the first line of the file (e.g. column headers).

mean = data.mean()
std_dev = data.std()

# Insert your code here
anomalous_samples = data[(data < mean - 2 * std_dev) | (data > mean + 2 * std_dev)]

''' This line identifies array elements that are outside the range of 2 standard deviations from the mean.
Finds elements that are less than the lower threshold and greater than the upper threshold.
' | ' --> OR operator
data[...]: Filters array elements that match the mask.'''

print(f"Found {len(anomalous_samples)} of atypical gene expression samples.")
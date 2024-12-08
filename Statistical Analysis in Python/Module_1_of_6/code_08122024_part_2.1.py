import pandas as pd
import matplotlib.pyplot as plt

# Loading data from CSV file
file = 'example_data_for_boxplot.csv'
df = pd.read_csv(file)
# print(df.head())

# Preparing box plot for all datases
plt.figure(figsize=(10, 6))
plt.boxplot([df['Set A'], df['Set B'], df['Set C'], df['Set D']])
'''
* creating a box plot for multiple data sets
* retrieves data from columns named Set A, Set B, Set C, Set D from the DataFrame object. Each column represents a data set to analyze.
'''

plt.title('Data sets comparison')
plt.xticks([1, 2, 3, 4], ['Set A', 'Set B', 'Set C', 'Set D'])
'''
* Sets the labels for the X-axis.
[1, 2, 3, 4]: Position numbers corresponding to each data set in the box plot.
['Set A', 'Set B', 'Set C', 'Set D']: Labels assigned to positions on the X-axis.
'''

plt.ylabel('Values')
plt.show()
# plt.savefig("fig.png")

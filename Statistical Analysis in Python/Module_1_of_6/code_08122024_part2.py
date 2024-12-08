import pandas as pd
import matplotlib.pyplot as plt

# Loading data from CSV file
file = 'example_data_for_boxplot.csv'
df = pd.read_csv(file)

# Choosing column for visualization
column_name = 'Set A'

# Preparing box plot
plt.figure(figsize=(8, 6))
plt.boxplot(df[column_name])
'''
* creates a new plot area;
* sets the figure size to 8x6 units (width x height).
* creates a box plot based on the given data.
    A box plot shows statistics of the data distribution:
    median, quartiles, minimum and maximum values;
'''

plt.title('Box plot for column: ' + column_name)  # adds a title and a Y-axis label;
plt.ylabel('Values')

plt.show()  # displays the plot in the GUI;
# plt.savefig("fig.png") # saves the current plot to a graphic file;

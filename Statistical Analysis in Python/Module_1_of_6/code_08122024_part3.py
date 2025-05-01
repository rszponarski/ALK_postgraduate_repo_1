import pandas as pd
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv('experimental_data.csv')

# Grouping data by fertilizer concentration
groups = df.groupby('Fertilizer concentration (%)')['Plant growth (cm)']
'''
Groups data by the values in the column "Fertilizer concentration (%)".
* ['Plant growth (cm)']: Extracts only the column "Plant growth (cm)"
'''

# Preparing the data for the chart
data_to_plot = [group for _, group in groups]
'''
* list comprehension;
*(_) Instead of giving a variable a name that will never be used, used underscore.
'''

# Creating a chart
plt.boxplot(data_to_plot)

# Ustawienie etykiet osi X
plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4'])

# Dodawanie tytułu i etykiet osi
plt.title('Plant growth depending on fertilizer concentration')
plt.xlabel('Fertilizer concentration')
plt.ylabel('Plant growth')

#plt.savefig("fig.png")
plt.show()
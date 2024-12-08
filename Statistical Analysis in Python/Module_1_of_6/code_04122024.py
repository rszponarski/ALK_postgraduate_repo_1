import pandas as pd

file = "example_data.csv"
df = pd.read_csv(file)

print("__Task1")
print(df.columns)
print("__Task2")
columns = ['Plant weight (g)', 'Number of leaves']  # create list
print(df[columns])
print("__Task3")
print(df.describe())
print("__Task4")
average = df[columns[0]].mean()
print(f'Mean value of {columns[0]} is {average}')
print("__Task4.v2")
column_name = 'Plant weight (g)'
average2 = df[column_name].mean()
print(f"Mean Value for {column_name}: {average2}")
print("__Task5")
column_name2 = 'Plant growth (cm)'
median = df[column_name2].median()
print(f'Median value of {column_name2} is equal to {median}')
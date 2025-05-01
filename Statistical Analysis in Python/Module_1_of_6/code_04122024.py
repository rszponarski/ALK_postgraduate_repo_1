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

print("__Task6")
column_name3 = "Number of leaves"
mode = df[column_name3].mode()
print(f"Mode of : {mode[0]}")
print(type(mode))
# the mode() method returns a Series Object
test_data1 = pd.Series([1, 1, 2, 3])  # creating two test data series
test_data2 = pd.Series([1, 1, 2, 2, 5, 5, 4])
print(f"Mode of data series {tuple(test_data1)} is: {test_data1.mode().tolist()}")
print(f"Mode of data series {tuple(test_data2)} is: {test_data2.mode().tolist()}")
'''
* pandas returns a Series object (along with indexes);
* This can be a single value (if there is only one most frequent value)
or multiple values 
* Pandas Series conversion:
    to tuple -> tuple()
    to list  -> series.tolist()
both used above'''

import pandas as pd

# TASK 1
file = 'atmospheric_conditions.csv'
data = pd.read_csv(file)

columns = data.columns
#print(columns)

for column in columns:
    max_value = data[column].max()
    min_value = data[column].min()

    range_value = round(max_value - min_value, 2)
    print(f"Value range for {column}: {range_value}")

# TASK 2
print("-----")

file = 'plants_height.csv'
data = pd.read_csv(file)

columns = data.columns

for column in columns:
    mean_value = round(data[column].mean(), 2)
    var_value = round(data[column].var(), 2)

    print(f"Column name: {column}")
    print(f"Mean value: {mean_value}")
    print(f"Variation: {var_value}")
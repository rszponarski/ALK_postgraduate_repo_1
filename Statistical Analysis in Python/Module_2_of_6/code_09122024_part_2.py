import pandas as pd

# TASK 4
# Reading go data
file_path = 'experimental_data.csv'
df = pd.read_csv(file_path)
columns_names = df.columns

std_list = []
for column in columns_names:
    mean_value = round(df[column].mean(), 1)
    max_value = df[column].max()
    min_value = df[column].min()
    range_value = round(max_value - min_value, 1)
    std_value = round(df[column].std(), 1)

    std_list.append(std_value)  # after each iteration we add the standard deviation value to the list;

    print(column)
    print(f"Mean: {mean_value}")
    print(f"Range: {range_value}")
    print(f"Standard deviation: {std_value}\n")

max_std = max(std_list)  # we find the maximum value of the list;
index_of_max = std_list.index(max_std)  # we find the index of the maximum value
print(f"Chosen dataset: {index_of_max + 1}")

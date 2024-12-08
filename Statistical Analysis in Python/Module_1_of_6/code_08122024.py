import pandas as pd

file = "experimental_data.csv"
df = pd.read_csv(file)

print(df.columns)  # display columns name
print('_________________')
print(df.head())  # display first 5 rows from DataFrame
print('_________________')

grouped_data = df.groupby("Plant growth (cm)")
''' NOTE:
    * data.groupby("Fertilizer concentration (%)") groups the data based on the value in the column
    "Fertilizer concentration (%)".
    * All rows that have the same value in that column will be grouped together.
'''

print(grouped_data)
print('''
The result of the above print() means that the grouped_data object is a DataFrameGroupBy object
and it does not display the data itself, but a representation of the object itself.
To display it, use a loop.''')

for name, group in grouped_data:
    median = round(group["Plant growth (cm)"].median(), 2)
    mean = round(group["Plant growth (cm)"].mean(), 2)
    mode = round(group["Plant growth (cm)"].mode()[0], 2)

    print(f"Fertilizer concentration: {name}%")
    print(f"Average height: {mean} cm")
    print(f"Median height: {median} cm")
    print(f"Growth mode: {mode} cm\n")

'''
* the group[] command is used in the context of GroupBy objects and is used to refer to columns in data groups;
* after performing a groupby(), you can use group[] to access a specific column within that group.

NOTE:
1. data.groupby("Fertilizer concentration (%)")
all rows that have the same value in the selected column will be grouped together.

2. for name, group in grouped_data:
'for loop' going through each group

The for loop iterates over the grouped_data object, which contains the data groups
- name - the name of each group - first we iterate over the values which we have chosen to group
e.g. group["Plant growth (cm)"] selects the "Plant growth (cm)" column from the given group
'''
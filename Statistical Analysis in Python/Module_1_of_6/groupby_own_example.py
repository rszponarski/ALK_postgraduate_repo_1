import pandas as pd

data = {'Plant growth (cm)': [5, 6, 5, 7, 8, 6, 5],
        'Fertilizer concentration (%)': [10, 10, 10, 20, 20, 30, 30]}

df = pd.DataFrame(data)
print(df)

print('After groupby() method: _____')
grouped_data = df.groupby('Fertilizer concentration (%)')
print('_____')

print('')
for name, group in grouped_data:
    print(f"Fertilizer concentration: {name}%")
    print(group)
    print("\n")

data2 = {'Plant growth (cm)': [5, 6, 5, 7, 8, 6, 5],
         'Fertilizer concentration (%)': [10, 10, 10, 20, 20, 30, 30],
         'Something else': [8, 14, 7, 1, 1, 1, 14]
         }

df2 = pd.DataFrame(data2)
grouped_data2 = df2.groupby('Fertilizer concentration (%)')

for name, group in grouped_data2:
    print(f"Fertilizer concentration: {name}%")
    print(group)
    print("\n")

'''
SUMMARY:
if you do not specify any columns in the for name, group in grouped data loop,
    then the loop will display all columns from the group (that is, all data that belongs to this group).'''

for name, group in grouped_data2:
    print(f"Fertilizer concentration: {name}%")
    print(group['Something else'])
    print("\n")

for name, group in grouped_data2:
    print(f"Concentration: {name}%")

    mode = group['Something else'].mode()[0]
    mean = round(group['Something else'].mean(), 2)

    print(f"Mode of 'Something else': {mode}")
    print(f"Mean of 'Something else': {mean}")
    print("\n")


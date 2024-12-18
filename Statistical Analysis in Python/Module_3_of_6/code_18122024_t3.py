# TASK 3 MODULE 3
# Verification of the null hypothesis for the two-sided alternative hypothesis
import pandas as pd
from scipy.stats import ttest_ind

data = pd.read_csv('chemical_data.csv')

area1 = data[data['Area'] == 'Area1']['Nitrate concentration']
area2 = data[data['Area'] == 'Area2']['Nitrate concentration']
''' Look at the data structure in the csv file:
    Nitrate concentration,  Area
    21.066330577056497,     Area1
    19.540502971191327,     Area1
    19.553256626883663,     Area1
    21.32877860223388,      Area1 etc.
--> data['Area'] selects the "Area" column from the DataFrame, which contains information about which area
(e.g. "Area1", "Area2", etc.) a given row belongs to.
--> data['Area'] == 'Area1' creates a logical series (i.e. a list of True/False values) that checks if each row
in the "Area" column corresponds to "Area1". If so, then it will be True for that row, otherwise False.

After narrowing the data to rows that have "Area 1" or "Area 2" in the "Area" column,
we select only the "Nitrate concentration" column to work with only that data.      SEE PRINT BELOW
'''
print(area1.head())


statistic, p_value = ttest_ind(area1, area2, equal_var=False)
''' Student's t-test:
    statistic, p_value = ttest_ind(area1, area2, equal_var=False)
--> ttest_ind performs a t-test for two samples: area1 and area2.
--> equal_var=False means that we assume that the variances in the two groups may differ,
    i.e. the test does not assume equal variances (so-called Welch's test).

    The function returns two values:
--> statistic: the value of the t-test statistic, which measures how big the difference is between the means
    of the two samples in relation to the variability in these samples.
--> p_value: the p-value, which is used to assess the statistical significance of the results.

Student's t-test is used to compare the means of two groups to see if the difference between those means
is statistically significant. Example: You want to see if the mean concentration of nitrate in Area1
is different from the mean concentration in Area2.  <-------------
'''

alpha = 0.05
print(f"Statystyka testowa: {statistic}")
print(f"Wartość p: {p_value}")

if p_value < alpha:
    print("Odrzucamy hipotezę zerową - istnieją istotne statystycznie różnice między obszarami.")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej - brak istotnych statystycznie różnic między obszarami.")

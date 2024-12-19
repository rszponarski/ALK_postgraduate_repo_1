# TASK 5 MODULE 3 -- ANOVA TEST
import pandas as pd
from scipy.stats import f_oneway

data = pd.read_csv('nitrate_data.csv')

# Data is filtered based on values in the Area column.
area1 = data[data['Area'] == 'Area1']['Nitrate_Concentration']
area2 = data[data['Area'] == 'Area2']['Nitrate_Concentration']
area3 = data[data['Area'] == 'Area3']['Nitrate_Concentration']

statistic, p_value = f_oneway(area1, area2, area3)
# The f_oneway function performs an ANOVA test.

alpha = 0.05  # significance level, which defines the limit for rejecting the null hypothesis (usually 0.05, or 5%).
print(f"Statystyka testowa: {statistic}")
print(f"Wartość p: {p_value}")

if p_value < alpha:  # If the p-value is less than the significance level (alpha): We reject the null hypothesis.
    print("Odrzucamy hipotezę zerową - istnieją statystycznie istotne różnice w średnich stężeniach azotanów.")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej - brak statystycznie istotnych różnic w średnich stężeniach"
          "azotanów.")

''' NOTES:

** What is ANOVA? **
--> ANOVA (Analysis of Variance) is a statistical method used to compare means across more than two groups
    to see if there are statistically significant differences between them.

** Why do we use ANOVA? **
--> If we wanted to compare means across three groups using the Student t-test, we would need to run three separate
    tests (Area1 vs. Area2, Area1 vs. Area3, Area2 vs. Area3).
--> ANOVA allows us to clearly and once test differences across multiple groups without increasing
    the risk of Type I error (false rejection of H₀).

** When to use ANOVA? **
    - When we want to see if the means across three or more groups are statistically significantly different.
    - When the data meets the assumptions of normality and homogeneity of variance.

-> HOMOGENEITY OF VARIANCES is an assumption in statistics that means that the variances in different groups
(e.g., samples) are the same or very close. In other words, the dispersion of data around the means in each
group should be similar.
'''
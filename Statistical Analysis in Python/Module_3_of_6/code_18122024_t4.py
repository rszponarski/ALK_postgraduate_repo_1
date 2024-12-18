# TASK 4 MODULE 3
# Verification of the null hypothesis for a one-sided alternative hypothesis

import pandas as pd
from scipy.stats import ttest_ind


data = pd.read_csv('chemical_data.csv')

area1 = data[data['Area'] == 'Area1']['Nitrate concentration']
area2 = data[data['Area'] == 'Area2']['Nitrate concentration']

statistic, p_value = ttest_ind(area1, area2, alternative="greater")

alpha = 0.05
print(f"Statystyka testowa: {statistic}")
print(f"Wartość p: {p_value}")

if p_value < alpha:
    print("Odrzucamy hipotezę zerową - istnieją statystycznie istotne różnice, średnie w obszarze drugim są większe.")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej - brak statystycznie istotnych różnic,"
          "średnie w obszarze drugim nie są większe.")
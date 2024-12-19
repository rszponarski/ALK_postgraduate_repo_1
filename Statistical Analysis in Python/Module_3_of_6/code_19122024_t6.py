# TASK 6 MODULE 3 -- PEARSON CORRELATION

import pandas as pd
from scipy.stats import pearsonr

data = pd.read_csv('nitrate_rainfall_data.csv')

# Calculating the Pearson correlation coefficient for two variables
correlation_coefficient, p_value = pearsonr(data["Nitrate_Concentration"], data["Rainfall"])
#   -> The pearsonr function calculates the Pearson correlation coefficient for two variables (in this case, nitrate
#   concentration and rainfall)
#   -> and the p-value of a statistical test to check whether the correlation is statistically significant.

print(f"Współczynnik korelacji Pearsona: {correlation_coefficient}")
print(f"Wartość p (p-value): {p_value}")

if abs(correlation_coefficient) >= 0.7:
    print("Istnieje silna korelacja liniowa.")
elif 0.3 <= abs(correlation_coefficient) < 0.7:
    print("Istnieje umiarkowana korelacja liniowa.")
else:
    print("Występuje słaba korelacja liniowa lub jej brak.")

'''
NOTES:
-> Pearson correlation is a measure of how strongly two variables are linearly related.
-> The value of the Pearson correlation coefficient (denoted r) ranges from -1 to 1, where:
    r=1     indicates a complete positive correlation (when one variable increases, the other also increases),
    r=−1    indicates a complete negative correlation (when one variable increases, the other decreases),
    r=0     indicates no linear correlation (the variables are independent).
    
The closer the value to 1 or −1, the stronger the correlation.
Values close to 0 indicate weak or no linear correlation.
'''

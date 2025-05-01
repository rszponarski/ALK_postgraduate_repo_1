# TASK 7 MODULE 3: Spearman correlation
import pandas as pd
from scipy.stats import spearmanr

data = pd.read_csv('nitrate_sunshine_data.csv')

correlation_coefficient, p_value = spearmanr(data["Nitrate_Concentration"], data["Sunshine_Hours"])

print(f"Współczynnik korelacji Spearmana: {correlation_coefficient}")
print(f"Wartość p (p-value): {p_value}")

if abs(correlation_coefficient) >= 0.7:
    print("Istnieje silna korelacja monotoniczna.")
elif 0.3 <= abs(correlation_coefficient) < 0.7:
    print("Istnieje umiarkowana korelacja monotoniczna.")
else:
    print("Występuje słaba korelacja monotoniczna lub jej brak.")

'''
RESULT:
A p-value of 0.28 is significantly greater than the typical significance level of 0.05.
This means that we do not have enough evidence to reject the null hypothesis that there is no monotonic correlation
between the variables.

NOTES:
Spearman correlation is a nonparametric measure that indicates the strength
and direction of a monotonic relationship between two variables.
It is used when the data do not meet the assumptions of Pearson correlation
(e.g., they are nonlinear, not normally distributed, or on an ordinal scale).

Spearman correlation calculates the coefficient rs, which takes values from -1 to 1:
rs=1:   perfect increasing monotonic relationship,
rs=−1:  perfect decreasing monotonic relationship,
rs=0:   no monotonic relationship.

Interpretation of the correlation coefficient:

If
∣rs∣ >= 0.7: Strong monotonic correlation.
0.3 =< ∣rs∣ =< 0.7: Moderate monotonic correlation.
|rs∣ < 0.3: Weak or no monotonic correlation.

___Major Differences Between Spearman and Pearson Correlation___
* Nature of Relationship:
-> Pearson correlation measures linear relationship between variables.
-> Spearman correlation measures monotonic relationship (not necessarily linear).

* Type of Data:
-> Pearson correlation requires quantitative data and normality of distribution.
-> Spearman correlation can be used for ordinal or nonlinear data.

* Robustness to Outliers:
-> Spearman correlation is more robust to outliers than Pearson correlation
because it is based on the ranks of the data.
'''
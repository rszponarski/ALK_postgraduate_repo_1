# TASK 8 MODULE 3

import pandas as pd
from scipy.stats import linregress

data = pd.read_csv('nitrate_rainfall_data.csv')

regression_result = linregress(data['Rainfall'], data['Nitrate_Concentration'])
slope = regression_result.slope
intercept = regression_result.intercept

'''
The code performs a linear regression analysis between two variables: Rainfall and Nitrate_Concentration. This analysis
determines whether a linear relationship exists between these variables and, if so, its strength and direction.
'''

print("Wyniki regresji liniowej:")
print(f"Nachylenie (nachylenie linii regresji): {slope}")
print(f"Punkt przecięcia (punkt przecięcia y linii regresji): {intercept}")
print(f"Współczynnik determinacji ((R-squared)): {regression_result.rvalue**2}")
print(f"Istotność statystyczna (p-value): {regression_result.pvalue}")

'''
The linregress function from the scipy.stats module is used to perform linear regression.
Linear regression is a statistical method that allows you to find the best-fit linear line to a set of data,
expressing the relationship between two variables:
    -X (independent variable),
    -Y (dependent variable).

Regression line formula:
    Y = slope * X + intercept

Arguments of the linregress function
    linregress(x, y)

-> Slope: Determines how much the dependent variable Y (nitrate concentration) changes with a unit change
in the independent variable X (precipitation).
-> Intercept: The value of Y when X=0.

* Calculation of the coefficient of determination:
regression_result.rvalue**2
Coefficient of determination (R^2): A measure of how well the regression line explains the variability in the data.
Values close to 1 indicate a good fit between the data and the model.

* Statistical Significance Result:
regression_result.pvalue
P-value: Tells whether the relationship between variables is statistically significant. Usually, if p < 0.05,
it means a significant relationship. If p < 0.05, we reject the null hypothesis (no relationship).
'''
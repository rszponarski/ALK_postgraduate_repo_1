# TASK 1 MODULE 3 part 2
import pandas as pd
import scipy.stats as stats

file = 'example_data.csv'
column_name = 'Nitrate concentration'
df = pd.read_csv(file)

mean_estimate = df[column_name].mean()
confidence_interval = stats.norm.interval(0.95, loc=mean_estimate, scale=stats.sem(df[column_name]))

print(f"Mean estimate for {column_name}: {mean_estimate:.2f}")
print(f"Confidence interval for Nitrate concentration: [{float(confidence_interval[0]):.3f},"
      f"{float(confidence_interval[1]):.3f}]\n")

# Step-by-step procedure
# 1- mean()
#           mean = data[column_name].mean()
# 2  - CI- Confidence Interval = Przedział ufności
#           CI = mean ± MOE
# 2.1- MOE- Margin of error = margines błędu
#           MOE = z * SEM
# 2.2- SEM- Standard Error of the Mean = Standardowy błąd średniej
#           SEM = Standard deviation / (n ** 0.5)
#          "n" is the number of samples.
# 2.3- z- Critical value = Wartość krytyczna
#           z_score = scipy.stats.norm.ppf(%_value)
# Critical values for the normal distribution can be found in statistical tables or using statistical functions (above);
# z_score = how many standard deviations do we need to move to get a certain level of confidence?
# 2.4- Standard Deviation = Odchylenie Standardowe
#   SD = a measure of the dispersion of data around the arithmetic mean
#   With PANDAS module
# Sample standard deviation (default ddof=1)    # Delta Degrees of Freedom = ddof
#      sample_std = data.std()
# Population standard deviation (ddof=0)
#      population_std = data.std(ddof=0)
# Calculating the number of elements in a column:
#      a) len(df['Column']) --> the number of all elements in the column (including NaN).
#      b) df['Column'].count() --> the number of non-empty (non-NaN) values.
#      c) df.shape[0] --> the number of all rows in the DataFrame.

no_elements = len(df['Nitrate concentration'])
print(f"Number of elements in the column: {no_elements}")
# sample_std = df.std()
# print(f"Standard Deviation (sample): {sample_std}")
population_std = df['Nitrate concentration'].std()
print(f"Standard Deviation (population): {population_std:.3f}")
z_score_min = stats.norm.ppf(0.025)
print(f"Z_score_min: {z_score_min:.2f}")
z_score = stats.norm.ppf(0.975)
print(f"Z_score_max: {z_score:.2f}")
SEM = population_std / (no_elements ** 0.5)
print(f"SEM: {SEM:.3f}")
MOE = z_score * SEM
print(f"MOE: {MOE:.3f}")
mean_ = df['Nitrate concentration'].mean()
print(f"Arithmetic mean: {mean_:.2f}")
CI = mean_ - MOE
CI_ = mean_ + MOE
print(f"Cofidence Interval: {float(CI):.3f} & {float(CI_):.3f}")

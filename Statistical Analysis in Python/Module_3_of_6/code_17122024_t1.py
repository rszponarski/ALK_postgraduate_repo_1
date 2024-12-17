# TASK 1 MODULE 3
import pandas as pd
import scipy.stats as stats

file = 'example_data.csv'
column_name = 'Nitrate concentration'
# A variable containing the name of the column for which we will analyze the data (nitrate concentration)
data = pd.read_csv(file)

mean_estimate = data[column_name].mean()
# Calculating the arithmetic mean of the selected column
confidence_interval = stats.norm.interval(0.95, loc=mean_estimate, scale=stats.sem(data[column_name]))
# Calculating the confidence interval       confidence level, loc=location parameter, scale=SEM

print(f"Mean estimate for {column_name}: {mean_estimate:.2f}")
print(f"Confidence interval for {column_name}: {list(confidence_interval)}")
print("-----explanation")
print(f"SEM: {stats.sem(data[column_name])}")
z_critical = stats.norm.ppf(0.975)  # 5%/2=0.025 -> 1-0.025=0.975
print(z_critical)  # result: 1.959963984540054
print(mean_estimate - z_critical * stats.sem(data[column_name]))
print(mean_estimate + z_critical * stats.sem(data[column_name]))

'''
NOTES:
______What is a confidence interval?
A confidence interval defines a range of values within which the true population mean lies with a specified
confidence level (e.g. 95%).

______The formula for a confidence interval for the normal distribution:
        Confidence interval = x_mean ± z * SEM
    
    x_mean: arithmetic mean
    z: The critical value for the normal distribution corresponding to the confidence level (for 95% it is 1.96).
-- The value 1.96 is the so-called z-score for a 95% confidence level.
-- The z-score tells us how many standard deviations from the mean we need to move to obtain a specified
confidence level.
-- The value of 1.96 can be found from the inverse function for the normal distribution
(quantile of the normal distribution):
-- The critical values for the normal distribution can be found in statistical tables or using statistical
functions (e.g. scipy.stats.norm.ppf).

______SEM: Standard error of the mean, calculated as:
        SEM = Standard deviation/ (n^0.5)
    "n" is the number of samples.
'''
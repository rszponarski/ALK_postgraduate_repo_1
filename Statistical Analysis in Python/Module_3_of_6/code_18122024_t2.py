# TASK 2 MODULE 3
import pandas as pd
import scipy.stats as stats

file = 'plant_data_park.csv'
column_name = 'Birches'
data = pd.read_csv(file)

mean = data[column_name].mean()
# TASK: Calculate a confidence interval
confidence_interval = stats.t.interval(0.9, len(data[column_name])-1, loc=mean, scale=stats.sem(data[column_name]))
# Calculating the CI           confidence level, degree of freedom, loc=location parameter, scale=SEM

# here            --> stats.t.interval
# previous task   --> stats.norm.interval;  EXPLANATION BELOW

print(len(data[column_name])-1)
# Reducing the number of samples by one, i.e. using n−1, follows from the concept of degrees of freedom in statistics.

print(f"Confidence interval for {column_name}: {confidence_interval}")
print("THE TRUE AVERAGE HEIGHT OF BIRCHES WITH A CERTAIN ASSUMED PROBABILITY IS IN THIS RANGE")

'''
NOTES:
Calculating a confidence interval:
    ___stats.t.interval: A function from the scipy.stats module to calculate confidence intervals
    based on the Student's t-distribution.
    ___0.9: Confidence level (90%), meaning that we are 90% certain that the true mean value lies
    within the calculated interval.
    ___len(data[column_name])-1: Degrees of freedom (number of observations minus 1).
    ___loc=mean: Center of the distribution (here the mean of the data).
    ____scale=stats.sem(data[column_name]): Standard error of the mean (SEM), calculated as the standard deviation divided by the square root of the number of observations.

!!! SEE:
The function calculates the limits of the confidence interval based on the Student's t-statistic,
which is appropriate for smaller samples (n < 30) or when the population standard deviation is unknown. !!!

*** Student's t-distribution:
-- It has "fatter tails" than the normal distribution, meaning that for small samples, extreme value are
    more likely to occur.
-- As the number of observations (degrees of freedom) increases, the Student's t-distribution approaches
    the normal distribution.

*** Differences between stats.norm.interval and stats.t.interval ***
--> Both are used to calculate confidence intervals, but their applications are different:

    1. stats.norm.interval (for normal distribution):
Assumes that the data comes from a population with a known standard deviation;
It is used when the sample size is large (n>30) or when the exact population parameters are known.
Confidence interval formula:

    CI = mean ± z * ( StandDevPopulation / n ** 0.5)

    2. stats.t.interval (for Student's t-distribution):
Assumes that the population standard deviation (is unknown, so we use the sample standard deviation;
Uses the Student's t-distribution, which is more appropriate for small samples.
Confidence interval formula:

    CI = mean ± t * ( StandDevSample / n ** 0.5)
    
    -> t - critical value of the t-Student distribution depending on the degrees of freedom (df=n−1),
    -> StandDevSample - standard deviation of the sample.

__Key difference:
stats.norm.interval assumes that the sample comes from a population with a known standard deviation
    and is based on the normal distribution.
stats.t.interval is more general because it assumes no knowledge of the population standard deviation
    and uses the sample standard deviation and the Student's t-distribution, making it more appropriate
    for small samples.

When to use which function?
-> stats.norm.interval:
    * When the population standard deviation is known.
    * When the sample is large (n>30).
-> stats.t.interval:
    * When the population standard deviation is unknown.
    * When the sample is small (n<=30).
'''
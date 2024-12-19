# TASK 4 MODULE 3
# Verification of the null hypothesis for a one-sided alternative hypothesis

import pandas as pd
from scipy.stats import ttest_ind


data = pd.read_csv('chemical_data.csv')

area1 = data[data['Area'] == 'Area1']['Nitrate concentration']
area2 = data[data['Area'] == 'Area2']['Nitrate concentration']

statistic, p_value = ttest_ind(area1, area2, alternative="greater")
# previous task -> the only difference in:
#                  = ttest_ind(area1, area2, equal_var=False)

alpha = 0.05  # SETS THE P-VALUE ACCEPTANCE THRESHOLD
# -> is used to establish the significance level of the test, i.e., it sets the threshold below which we reject
#    the null hypothesis.
# -> This does not directly indicate a difference between the means, but rather sets a decision criterion based on
#    the p-value values.

print(f"Statystyka testowa: {statistic}")
print(f"Wartość p: {p_value}")

if p_value < alpha:
    print("Odrzucamy hipotezę zerową - istnieją statystycznie istotne różnice, średnie w obszarze drugim są większe.")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej - brak statystycznie istotnych różnic,"
          "średnie w obszarze drugim nie są większe.")

''' NOTES:
    jednostronna hipoteza alternatywna                         obustronna hipoteza alternatywna
ttest_ind(area1, area2, alternative="greater")      VS.     ttest_ind(area1, area2, equal_var=False)

--> the difference in this code from the previous one is the use of the alternative="greater" parameter in the ttest_ind
function. This modifies the way hypotheses are formulated in the Student's t-test, changing the test from two-sided
to one-sided.
--> By default, the Student's t-test tests a two-sided hypothesis.

* PREVIOUS TASK:
_____ ttest_ind(area1, area2, equal_var=False)
-> H0 (null hypothesis): The mean of the two groups is equal (mean_1 = mean_2)
-> H1 (alternative hypothesis): The means of the two groups are different (mean_1 = mean_2).

* CURRENT TASK:
(one-sided test with alternative="greater")
Using the alternative="greater" parameter modifies the hypotheses:
-> H0 (null hypothesis): The mean of the first group is greater than or equal to the mean of the second group
    (mean_1 >= mean_2)
-> H1 (alternative hypothesis): The mean of the second group is greater than the mean of the first group
    (mean_1 < mean_2)
    A one-sided test examines only one possibility of difference (the mean in Area1 is greater than the mean in Area1).

* Implications of the difference:
A two-sided test (the default) is more general because it tests whether there is any difference between the means,
    regardless of direction.
A one-sided test is more specific because it tests the difference in one specific direction
    (e.g. whether the mean in Area1 is greater).

* Why did we use Welch's test in the previous task?
- equal_var=False in the previous code forced us to use Welch's test.
Reason: We were not sure whether the variances of both groups (for Area1 and Area2) were Area2) are equal.

* Why don't we use equal_var anymore?
Previous approach:
- equal_var=True (default in old versions of SciPy): It assumed that the variances of both groups were equal
    (so-called classic Student's t-test).
- equal_var=False: Used when the variances of the groups were different (Welch's test).

* New approach:
In newer versions of SciPy, the Welch test is the default and more versatile.
We don't have to manually specify the equal_var parameter because:

* The Welch test performs better when the variances are different.
It is considered to be more statistically robust in the case of unequal variances.
* If equal_var=True is not explicitly specified, the function automatically uses the Welch test.
equal_var=False IS SET BY DEFAULT
<If you are using Scipy version 1.6 or later, equal_var=False is the default>
'''
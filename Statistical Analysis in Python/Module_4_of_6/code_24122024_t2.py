# TASK 2 MODULE 4
import pandas as pd
from sklearn.preprocessing import StandardScaler
# StandardScaler: A tool from the scikit-learn library that allows to standardize data

# Data reading
df = pd.read_csv('pizza.csv').drop(['brand', 'id'], axis=1)

# Initializing the scaler
scaler = StandardScaler()
''' -> create a scaler object that standardizes the data.
Standardization transforms each column of data so that:
    * The mean of each column is 0.
    * The standard deviation of each column is 1. '''

pizza_scaled = scaler.fit_transform(df)
'''
1. fit: Calculates the mean and standard deviation for each column in df.
2. transform: Transforms the data in each column according to the formula:
        z = (x−μ) / σ
where:
    x: Original value in the column.
    μ: Mean of the column.
    σ: Standard deviation of the column.
The result is a pizza_scaled matrix where all columns are standardized.'''

scaled_df = pd.DataFrame(pizza_scaled, columns=df.columns)
''' Create a new scaled_df data frame, where each column corresponds to a column from the original df frame,
but the values are already standardized.'''

# Basics statistics printing
print(scaled_df[['prot', 'fat', 'carb']].describe())
''' We select the prot, fat, and carb columns from the new data frame.
.describe():    Calculates basic statistics for these columns:
    count:          The number of observations in each column.
    mean:           The mean (for standardized data, this should be close to 0).
    std:            The standard deviation (for standardized data, this should be close to 1).
    min and max:    The minimum and maximum values.
    25%, 50%, 75%:  Quartiles of the data.
'''
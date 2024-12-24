# TASK 2 MODULE 4
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Data reading
df = pd.read_csv('pizza.csv').drop(['brand', 'id'], axis=1)

scaler = StandardScaler()
pizza_scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(pizza_scaled, columns=df.columns)

# Basics statistics printing
print(scaled_df[['prot', 'fat', 'carb']].describe())
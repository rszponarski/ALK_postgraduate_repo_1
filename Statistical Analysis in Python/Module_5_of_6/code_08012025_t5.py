# TASK 5 MODULE 5

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Loading a sample dataset and omitting the id cluster
data = pd.read_csv('pizza.csv').drop('id', axis=1)
X = data.drop('brand', axis=1)
y = data['brand']

# Performing the t-SNE transformation
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Creation of a new data set containing the 'brand' column
df = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
df['brand'] = y

# Data visualisation after dimensionality reduction
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='t-SNE 1', y='t-SNE 2', hue='brand')
plt.title('t-SNE - visualisation with colouring by brand')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig('fig5.png')

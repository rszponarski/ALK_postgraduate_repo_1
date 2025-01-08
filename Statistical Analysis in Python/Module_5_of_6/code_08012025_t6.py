# TASK 6 MODULE 5
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# Loading data with the omission of id and brand columns
X = pd.read_csv('pizza.csv').drop(['id', 'brand'], axis=1)

# Performing the t-SNE transformation
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Selecting a feature to colour - for example, "ash"
feature_to_color = X['ash']

# Drawing a t-SNE diagram with colouring according to the selected characteristic
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=feature_to_color)
plt.colorbar(scatter)
plt.title('t-SNE visualisation with colouring by "ash" feature')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig('fig6.png')

# TASK 3 MODULE 5
import pandas as pd
# from scipy.stats import pointbiserialr # is not used
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Data reading
df = pd.read_csv('seeds_dataset.csv')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42) # <- up to this point the same as TASK 1
df['Clusters'] = kmeans.fit_predict(df_scaled)
'''The first code uses df['Clusters'], which means that the classified clusters are added directly as a column to 
the DataFrame (df). The code from Task 1 uses the clusters variable instead, which stores only the predicted classes
(clusters), and the DataFrame (df) itself is left without an additional column.

    Area  Perimeter  ...  Length of kernel groove  Clusters
0  15.26      14.84  ...                    5.220         2
1  14.88      14.57  ...                    4.956         2
'''

g = sns.pairplot(df, hue='Clusters', palette='Set2', height=1.6, aspect=1.4, plot_kws={'s':7})
g.add_legend()
plt.savefig('fig3.png')
#print(df.head())

''' NOTES:
sns.pairplot(df, hue='Clusters', ...) – generuje wykres paryplotu, gdzie każdy wykres pokazuje zestaw współczynników
z kolorami reprezentującymi klastry.

__hue='Clusters' – kolumny z oznaczeniami klastrów są używane do różnicowania kolorów na wykresie.
__palette='Set2' – używany kolor dla klastrów (można dostosować paletę do swoich preferencji).
__height=1.6 – wysokość wykresu.
__aspect=1.4 – proporcje wysokości i szerokości wykresu.
__plot_kws={'s':7} – parametry plotowania (rozmiar punktów).
__g.add_legend() – dodaje legendę do wykresu.
__plt.savefig('fig3.png') – zapisuje wygenerowany wykres jako plik PNG o nazwie fig3.png.

klasy = klastry -> grupy podobnych danych'''
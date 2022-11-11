import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

df = pd.read_csv("dataset.csv",header=None)
a = []

X = df.copy()

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    a.append(kmeans.inertia_)
plt.plot(range(1,11),a)
plt.title("Graph to determine Epsilon Value for DBSCAN")
plt.xlabel("Clusters")
plt.show("Array for Cluster sum")

db = DBSCAN(eps=0.91, min_samples=10)
labels = db.fit_predict(X)


plt.scatter(X.iloc[labels == -1, 0], X.iloc[labels == -1, 1], c='black', label='Outliers')
plt.scatter(X.iloc[labels == 0, 0], X.iloc[labels == 0, 1], c='blue', label='Cluster 1')
plt.scatter(X.iloc[labels == 1, 0], X.iloc[labels == 1, 1], c='blue')
plt.scatter(X.iloc[labels == 2, 0], X.iloc[labels == 2, 1], c='red')
plt.scatter(X.iloc[labels == 3, 0], X.iloc[labels == 3, 1], c='red', label='Cluster 2')
plt.scatter(X.iloc[labels == 4, 0], X.iloc[labels == 4, 1], c='red')
plt.scatter(X.iloc[labels == 5, 0], X.iloc[labels == 5, 1], c='red')
plt.legend(loc='best')
plt.title("DBSCAN Clustering Plot")
plt.xlabel("x values from file")
plt.ylabel("y values from file")
plt.show()




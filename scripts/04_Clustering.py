import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# For demonstration, using sample data
X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)
print(f'K-Means Silhouette Score: {kmeans_silhouette:.4f}')

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(X)
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(X, dbscan_labels)
    print(f'DBSCAN Silhouette Score: {dbscan_silhouette:.4f}')
else:
    print('DBSCAN found only one cluster or noise')

# Visualize clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.show()

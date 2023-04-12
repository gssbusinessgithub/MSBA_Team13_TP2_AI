# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 02:44:44 2023

@author: gsste
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
X = np.random.randn(100, 2)

# Cluster the data using k-means
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

# Plot the data and the cluster centers
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

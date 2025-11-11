# Step 0: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# ------------------------------
# Step 1: Load your dataset
# ------------------------------
df = pd.read_csv("your_data.csv")

# Quick look
print(df.head())
print(df.info())
print(df.describe())

# ------------------------------
# Step 2: Handle categorical features
# ------------------------------
# Example: "State" column
categorical_cols = ['State']  # Add more if needed
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# ------------------------------
# Step 3: Handle missing values (optional)
# ------------------------------
df_encoded = df_encoded.fillna(df_encoded.mean())  # simple imputation

# ------------------------------
# Step 4: Standardize numeric features
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# ------------------------------
# Step 5: PCA
# ------------------------------
# Reduce to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA scatter
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter Plot")
plt.show()

# ------------------------------
# Step 6: Isolation Forest (Outlier Detection)
# ------------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso.fit_predict(X_pca)

# -1 = outlier, 1 = inlier
mask_inliers = outlier_labels == 1
mask_outliers = outlier_labels == -1

print(f"Detected outliers: {np.sum(mask_outliers)} / {len(outlier_labels)}")

# Scatter plot highlighting outliers
plt.figure(figsize=(8,6))
plt.scatter(X_pca[mask_inliers,0], X_pca[mask_inliers,1], c='blue', label='Inliers', alpha=0.7)
plt.scatter(X_pca[mask_outliers,0], X_pca[mask_outliers,1], c='red', label='Outliers', marker='x')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA with Isolation Forest Outliers")
plt.legend()
plt.show()

# ------------------------------
# Step 7: K-means clustering
# ------------------------------
# Use only inliers for clustering
X_clean = X_pca[mask_inliers]

# Optional: use elbow method to find k
inertias = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clean)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Assume k=3 for demonstration
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_clean)

# Visualize clusters
plt.figure(figsize=(8,6))
plt.scatter(X_clean[:,0], X_clean[:,1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-means Clusters after PCA")
plt.show()

# ------------------------------
# Step 8: Attach cluster labels to original data
# ------------------------------
df_clusters = df_encoded.copy()
df_clusters['Cluster'] = np.nan
df_clusters.loc[mask_inliers, 'Cluster'] = cluster_labels
df_clusters['Outlier'] = outlier_labels

# ------------------------------
# Step 9: GroupBy cluster analysis
# ------------------------------
# Analyze mean of numeric features per cluster
cluster_summary = df_clusters.groupby('Cluster').mean()
print(cluster_summary)

# Optional: visualize cluster means
cluster_summary.plot(kind='bar', figsize=(10,6))
plt.title("Cluster Feature Means")
plt.show()

# Optional: categorical distribution per cluster
for col in categorical_cols:
    pd.crosstab(df_clusters['Cluster'], df[col]).plot(kind='bar', figsize=(8,5))
    plt.title(f"{col} distribution per cluster")
    plt.show()

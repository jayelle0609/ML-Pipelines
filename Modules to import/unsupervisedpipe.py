# ==============================================
# FULL UNSUPERVISED LEARNING PIPELINE
# ==============================================

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
# -------------------------------
# Example Dataset
# -------------------------------
df = pd.DataFrame({
    'age': [25, 32, 47, 51, 62],
    'income': [50000, 60000, 55000, 75000, 90000],
    'gender': ['M','F','F','M','F'],
    'region': ['North','South','East','West','North'],
    'education': ['high','medium','low','medium','high']
})
df = pd.read_csv('your_dataset.csv') 
# -------------------------------
# Column selection
# -------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
ordinal_cols = ['education']
education_order = ['low','medium','high']
categorical_cols = [c for c in categorical_cols if c not in ordinal_cols]

# -------------------------------
# Preprocessing Pipelines
# -------------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[education_order]))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
    ('ord', ordinal_pipeline, ordinal_cols)
])

# dont usually use pipeline for clustering, we use preprocessor.fit_transform directly

# -------------------------------
# Build Clustering Pipeline
# -------------------------------

df_processed = preprocessor.fit_transform(df)

results = []

for n_clusters in range(2, 12):
    print(f'Clustering with {n_clusters} clusters')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1, max_iter= 300) 

    #mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=42, max_iter=50, n_init=1, verbose =1) # verbose prints iteration updates
    #labels = mbk.fit_predict(train_processed)
    #inertia = mbk.inertia_

    # max_iter is the number of iterations for a single run of KMeans to converge the moving centroids till stabilize
    # n_init is the number of times KMeans algo runs with diff random centroid seeds in each cluster, best result (lowest inertia) is kept

    labels = kmeans.fit_predict(df_processed) # gives u the cluster labels
    inertia = kmeans.inertia_ # sum of squared distances to closest cluster center
    #sil = silhouette_score(train_processed, labels) # gives u the silhouette score, higher is better (how well separated the clusters are)
    ch = calinski_harabasz_score(df_processed, labels) # higher is better, ratio of between-cluster variance to within-cluster variance
    db = davies_bouldin_score(df_processed, labels) # lower is better, average similarity betw clusters
    results.append({
        'n_clusters': n_clusters,
        'inertia': inertia,
        #'silhouette_score': sil, # silhouteete score is computationally very expensive, takes forever to run
        # sil computes pairwise distances between all points, which is O(n²) complexity.
        'Calinski-Harabasz': ch,
        'Davies-Bouldin': db
    })

results_df = pd.DataFrame(results)
print(results_df)

# -------------------------------
# Optimal K selection with 2nd differential
# -------------------------------
results_df['inertia_second_diff'] = results_df['inertia'].diff().abs().diff().abs().round() 
# this gives the second derivative of inertia values, which tells us the rate of change of the rate of change
optimal_k_inertia = results_df.loc[results_df['inertia_second_diff'].idxmax(), 'n_clusters'] # idxmax gives the index of the max value in the series
print(f'Optimal number of clusters based on Inertia second derivative: {optimal_k_inertia}') 

# -------------------------------
# Elbow Method : Optimal K selection 
# -------------------------------
fig, axes = plt.subplots(2,2, figsize=(8, 8))
# set font size for all subplots and no overlap
plt.rcParams.update({'font.size': 5})

# Elbow Method Plot
axes[0,0].plot(results_df['n_clusters'], results_df['inertia'], marker='o')
axes[0,0].set_xlabel('Number of Clusters')
axes[0,0].set_ylabel('Inertia, SSD to Closest Cluster Center')
axes[0,0].set_title('Elbow Method for optimal k cluster')
# "Inertia is the sum of squared distances to the closest cluster center (centroid). 
# As the number of clusters increases, inertia decreases because data points are closer to their assigned centroids. 
# The 'elbow' point indicates an optimal number of clusters where adding more clusters yields diminishing returns in reducing inertia.", fontsize=5)

"""# Silhouette Score Plot
axes[0,1].plot(results_df['n_clusters'], results_df['silhouette_score'], marker='o', color='orange')
axes[0,1].set_xlabel('Number of Clusters')
axes[0,1].set_ylabel('Silhouette Score (how well separated clusters are)')
axes[0,1].set_title('Silhoutte Analysis for optimal k cluster')
# "The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. 
# A higher score indicates better-defined clusters. The plot helps identify the optimal number of clusters where the silhouette score is maximized."
"""

# Calinski-Harabasz Score Plot
axes[1,0].plot(results_df['n_clusters'], results_df['Calinski-Harabasz'], marker='o', color='green')
axes[1,0].set_xlabel('Number of Clusters')
axes[1,0].set_ylabel('Calinski-Harabasz Score (higher is better)')
axes[1,0].set_title('Calinski-Harabasz Analysis for optimal k cluster')
# The Calinski-Harabasz Score evaluates cluster validity by measuring the ratio of between-cluster variance to within-cluster variance. 
# A higher score indicates better-defined clusters. The plot helps identify the optimal number of clusters where this score is maximized."

# Davies-Bouldin Score Plot
axes[1,1].plot(results_df['n_clusters'], results_df['Davies-Bouldin'], marker='o', color='red')
axes[1,1].set_xlabel('Number of Clusters')
axes[1,1].set_ylabel('Davies-Bouldin Score (lower is better)')
axes[1,1].set_title('Davies-Bouldin Analysis for optimal k cluster')
# The Davies-Bouldin Score assesses cluster quality by measuring the average similarity ratio of each cluster with its most similar cluster. 
# A lower score indicates better-defined clusters. The plot helps identify the optimal number of clusters where this score is minimized.

# -------------------------------
# Run KMeans with optimal k
# -------------------------------

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, verbose=1, max_iter=300)
cluster_labels = kmeans.fit_predict(df_processed)

labels = kmeans.fit_predict(df_processed)
print(f'Cluster labels for optimal k={optimal_k}: {cluster_labels}')

# Add in cluster labels to original dataframe
df['cluster'] = labels
df

# -------------------------------
# Visualize clusters distribution (balanced?)
# -------------------------------
# Bar plot
plt.rcParams.update({'font.size': 12}) 
plt.figure(figsize=(3,3))
df['cluster'].value_counts().sort_index().plot(kind='bar', title='Cluster Counts')

# pie chart
cluster_counts = df['cluster'].value_counts().sort_index()
plt.figure(figsize=(3,3))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Cluster Distribution')
plt.show()

# -------------------------------
# Visualize Results - Heatmap of feature means per cluster
# -------------------------------

# map if necessary categorical columns to numeric for heatmap
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df

# Heatmap of feature means per cluster
numeric_cols = df.select_dtypes(include=np.number).columns
cluster_means = df.groupby('cluster')[numeric_cols].mean()
plt.figure(figsize=(10, 6))

# set font size
plt.rcParams.update({'font.size': 10})
sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Means per Cluster')
plt.xlabel('Features')

# feauture correlation heat map : -1 to 1
# cluster feature heatmap : shows mean value of each feature in each cluster
# feauture correlation heat map : -1 to 1
# cluster feature heatmap : shows mean value of each feature in each cluster

# -------------------------------
# Visualise : Bar plot of feature means by cluster
# -------------------------------
for cols in df.columns:
    plt.figure(figsize=(3,3))
    sns.barplot(x='cluster', y=cols, hue = 'cluster', data=df)
    plt.title(f'Boxplot of {cols} by Cluster')
    plt.show()

# -------------------------------
# Visualise : Scatter plot of 2 cols, hue by color
# -------------------------------

numeric_cols = df.select_dtypes(include='number').columns.tolist()

for x_col, y_col in combinations(numeric_cols, 2):
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='cluster', palette='Set1')
    plt.title(f'{y_col} vs {x_col} by Cluster')
    plt.show()

# -------------------------------
# Visualise : PCA (2D) plot of clusters
# -------------------------------

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['cluster']))
df_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=train['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster')
plt.title('PCA Visualization of Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()  

# -------------------------------
# Visualise : PCA (3D) plot of clusters
# -------------------------------

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

pca = PCA(n_components=3)
df_pca = pca.fit_transform(X_scaled))

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# Scatter points
ax.scatter(
    df_pca[:, 0], 
    df_pca[:, 1], 
    df_pca[:, 2], 
    c=df['cluster'],  # color by cluster
    cmap='tab10', 
    s=10
)

# Labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Clusters Visualization')

plt.show()

# -------------------------------
# Top 3 Features for each PC - Print & Barplot
# -------------------------------
# Each row = feature, each column = PC
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=df.drop(columns=['cluster']).columns
)

# Absolute values show magnitude of contribution
loadings_abs = loadings
print(loadings_abs)

# -------------------------------
# Print Top 3 features for each PC
# -------------------------------
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\nTop 3 features for {pc}:")
    print(loadings_abs[pc].sort_values(ascending=False).head(3))

# -------------------------------
# Barplot of Top 3 features for each PC
# -------------------------------

pc = ['PC1', 'PC2', 'PC3']

for i in range(3):
    plt.figure(figsize=(3,3))
    top_features = loadings_abs[pc[i]].sort_values(ascending=False).head(3)
    top_features.plot(kind='bar', color='skyblue', title=f'Top 3 Features for {pc[i]}')
    plt.ylabel('Absolute Contribution')
    plt.show()

# -------------------------------
# Heatmap of Feature Importance via PCA Loadings
# -------------------------------

plt.figure(figsize=(8,6))
sns.heatmap(loadings_abs, annot=True, cmap='coolwarm')
plt.title('Feature Importance per Principal Component')
plt.show()

# ==============================================
# STATISTICAL TESTS TO VALIDATE CLUSTERS ARE DIFFERENT SIGNIFICANTLY
# ==============================================

# -------------------------------# -------------------------------# -------------------------------
# Statistical test to check if clusters differ significantly on key features (numeric cols only)
# -------------------------------
from scipy.stats import f_oneway, kruskal

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    groups = [df[df['cluster']==i][col] for i in df['cluster'].unique()]
    
    # ANOVA
    f_stat, p_val = f_oneway(*groups)
    print(f"ANOVA for {col}: F={f_stat:.2f}, p={p_val:.4f}")

    # Non-parametric alternative
    h_stat, p_val_h = kruskal(*groups)
    print(f"Kruskal-Wallis for {col}: H={h_stat:.2f}, p={p_val_h:.4f}")
    print("-"*40)

# But anova is based on assumption of normal distrbution within each group.
# ANOVA Purpose: Tests whether feature means are statistically different across two or more groups (clusters). Or due to random noise.
# p<0.05 indicates significant difference in means across clusters.
# F-statistic = Ratio of between-group variance to within-group variance. 
# Higher → more separation, clusters are statistically different from each other.

# -------------------------------
# Statistical test to check if clusters differ significantly on key features (cat cols only)
# -------------------------------
from scipy.stats import chi2_contingency

categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
for col in categorical_cols:
    contingency = pd.crosstab(df['cluster'], df[col])
    chi2, p, dof, ex = chi2_contingency(contingency)
    print(f"Chi-square for {col}: chi2={chi2:.2f}, p={p:.4f}")

# Purpose: Tests whether a categorical feature is associated with cluster membership.
# p < 0.05 → reject H₀ → the feature is significantly associated with cluster membership.
# H0 : Feature is not associated with cluster membership.


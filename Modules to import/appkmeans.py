# unsupervised_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.stats import f_oneway, kruskal, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Unsupervised Learning App")
st.success("##### Upload a dataset to perform Clustering + PCA + Statistical Analysis.")

# -------------------------------
# Upload CSV or use sample
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default sample dataset")
    df = pd.read_csv("sample.csv")  # Ensure 'sample.csv' exists in folder

st.write("Data preview:")
st.dataframe(df.head())

# -------------------------------
# Drop unwanted columns
# -------------------------------
all_columns = df.columns.tolist()
cols_to_drop = st.multiselect("Select columns to drop (ex. irrelevant columns)", all_columns)
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    st.write(f"Dropped columns: {cols_to_drop}")
    st.write("Updated Data Preview:")
    st.dataframe(df.head())

# -------------------------------
# Identify numeric/categorical
# -------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
st.info(f"Numeric columns: {numeric_cols}")
st.info(f"Categorical columns: {categorical_cols}")

# -------------------------------
# Ordinal columns
# -------------------------------
st.write("## Specify Ordinal Columns")
st.caption("Specify categorical columns where the order matters (ex. education) and their specific order from lowest to highest (ex. ALevel, Bachelors, Masters).")

ordinal_cols = st.multiselect("Select Ordinal Columns", categorical_cols)
ordinal_orders = {}

for col in ordinal_cols:
    df[col] = df[col].astype(str).str.lower()
    unique_vals = df[col].unique()
    st.write(f"###### Unique values for '{col}': {unique_vals}")
    order_str = st.text_input(f"Enter order for **{col}** (comma-separated)", value=",".join(unique_vals))
    ordinal_orders[col] = [x.strip().lower() for x in order_str.split(",")]

# Remove ordinal columns from categorical_cols
categorical_cols = [c for c in categorical_cols if c not in ordinal_cols]

# -------------------------------
# Preprocessing info
# -------------------------------
st.subheader("Backend Pipeline Workflow")
st.write("In the backend, the following preprocessing and analysis steps are performed:")
st.info("""**Preprocessing the data:**
- Dropping any columns you chose
- Imputing missing values in numeric/categorical columns with mean and mode respectively.
- Scaling numeric features using **StandardScaler**
- Encoding categorical features using **OneHotEncoder**
- Encoding ordinal columns using **OrdinalEncoder** per specified order
""")

st.caption("""[EASY LAYMAN EXPLANATION](https://docs.google.com/document/d/13CilRy_dplJYhaXDSUkflicZhsNIM_j5voAo74__Y2o/edit?usp=sharing)""")
st.info("""**Preparing for KMeans and PCA:**
        
- Optimal k clusters evaluated (2–11) using **[Inertia](https://en.wikipedia.org/wiki/Elbow_method_(clustering))**, **[Calinski-Harabasz](https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index)**, and **[Davies-Bouldin scores](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index)**.
- **[Second derivative of inertia](https://franknielsen.github.io/HPC4DS-kmeans-Chapter7.pdf)** used to suggest optimal k
- PCA for 2D/3D cluster visualization
- Statistical tests (**ANOVA**, **Kruskal-Wallis**, **Chi-square**) to validate cluster differences
""")

# -------------------------------
# Pipeline
# -------------------------------
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                             ('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))])
ordinal_pipeline = None
if ordinal_cols:
    ordinal_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('ordinal', OrdinalEncoder(categories=[ordinal_orders[col] for col in ordinal_cols]))])

transformers = [('num', numeric_pipeline, numeric_cols),
                ('cat', categorical_pipeline, categorical_cols)]
if ordinal_cols:
    transformers.append(('ord', ordinal_pipeline, ordinal_cols))

preprocessor = ColumnTransformer(transformers)
df_processed = preprocessor.fit_transform(df)

# -------------------------------
# Optimal k evaluation
# -------------------------------
st.write("## Optimal Cluster Selection (k)")
k_range = list(range(2,12))
results = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_processed)
    results.append({'k':k,
                    'inertia':kmeans.inertia_,
                    'CH_score':calinski_harabasz_score(df_processed, labels),
                    'DB_score':davies_bouldin_score(df_processed, labels)})
results_df = pd.DataFrame(results)
results_df['inertia_second_diff'] = results_df['inertia'].diff().abs().diff().abs()
optimal_k_inertia = results_df.loc[results_df['inertia_second_diff'].idxmax(), 'k']
st.write(f"Suggested optimal k (second derivative of inertia): {optimal_k_inertia}")

# -------------------------------
# Plot elbow, CH, DB, 2nd derivative
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.rcParams.update({'font.size': 13})

# ---- Elbow Method ----
axes[0,0].plot(results_df['k'], results_df['inertia'], marker='o')
axes[0,0].axvline(optimal_k_inertia, color='red', linestyle='--')
axes[0,0].set_title("Elbow Method - Inertia")
axes[0,0].set_xlabel("k")
axes[0,0].set_ylabel("Inertia")

# ---- Calinski-Harabasz ----
axes[0,1].plot(results_df['k'], results_df['CH_score'], marker='o', color='green')
axes[0,1].set_title("Calinski-Harabasz Score")
axes[0,1].set_xlabel("k")
axes[0,1].set_ylabel("CH Score")

# ---- Davies-Bouldin ----
axes[1,0].plot(results_df['k'], results_df['DB_score'], marker='o', color='orange')
axes[1,0].set_title("Davies-Bouldin Score")
axes[1,0].set_xlabel("k")
axes[1,0].set_ylabel("DB Score")

# ---- Second Derivative ----
axes[1,1].plot(results_df['k'], results_df['inertia_second_diff'], marker='o', color='purple')
axes[1,1].axvline(optimal_k_inertia, color='red', linestyle='--')
axes[1,1].set_title("Second Derivative of Inertia")
axes[1,1].set_xlabel("k")
axes[1,1].set_ylabel("Second Derivative")

# ---- Explanations ----
fig.text(0.03, -0.03,
         "• Elbow method to identify optimal k clusters → where inertia stops decreasing sharply. \n"
         "• Mathematically, use 2nd differential to find optimal k. \n"
         "• Inertia is the sum of squared distances between each data point and its assigned cluster centroid → lower inertia indicates tighter clusters.\n"
         "• Calinski–Harabasz Score  → higher indicates better-defined clusters.\n"
         "• Davies–Bouldin Score → lower indicates more distinct clusters.",
         ha='left', va='center', fontsize=12, style='italic')
# CH = ratio of the between-cluster dispersion (or between-group sum of squares) 
# to the within-cluster dispersion (or within-group sum of squares), normalized by their degrees of freedom.


plt.tight_layout(rect=[0, 0.05, 1, 1])
st.pyplot(fig)

# -------------------------------
# Select k interactively
# -------------------------------
n_clusters = st.slider("Select number of clusters to use", 2, 11, optimal_k_inertia)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(df_processed)
df['cluster'] = labels

# -------------------------------
# Compute PCA
# -------------------------------
pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_processed)

# Helper to get feature names after preprocessing
def get_feature_names(preprocessor):
    feature_names = []
    if 'num' in preprocessor.named_transformers_:
        feature_names.extend(preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_cols))
    if 'cat' in preprocessor.named_transformers_:
        feature_names.extend(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))
    if 'ord' in preprocessor.named_transformers_ and ordinal_cols:
        feature_names.extend(preprocessor.named_transformers_['ord'].named_steps['ordinal'].get_feature_names_out(ordinal_cols))
    return feature_names

processed_feature_names = get_feature_names(preprocessor)
loadings = pd.DataFrame(pca.components_.T, index=processed_feature_names, columns=['PC1','PC2','PC3'])
loadings_abs = loadings.abs()

# -------------------------------
# Top 2x2: Cluster dist, top 3 PC features, heatmap
# -------------------------------
fig, axes = plt.subplots(2,2, figsize=(16,12))
plt.rcParams.update({'font.size':14})

# Cluster distribution
ax1 = axes[0,0]
cluster_counts = df['cluster'].value_counts().sort_index()
bars = ax1.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.magma(np.linspace(0.2,0.8,len(cluster_counts))))
ax1.set_title("Distribution of Clusters")
ax1.set_xlabel("Cluster"); ax1.set_ylabel("Count")
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height+0.5, f'{int(height)}', ha='center', va='bottom')

# Top 3 features by PC1
ax2 = axes[0,1]
top_n = 3
x = np.arange(top_n)
bar_width = 0.25
for i, pc in enumerate(['PC1','PC2','PC3']):
    top_features = loadings_abs[pc].sort_values(ascending=False).head(top_n)
    ax2.bar(x + i*bar_width, top_features.values, width=bar_width, label=pc)
    for j,val in enumerate(top_features.values):
        ax2.text(x[j]+i*bar_width, val+0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=12)
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(top_features.index, rotation=45, ha='right')
ax2.set_ylabel("Absolute Contribution to each PC component"); ax2.set_title("Top 3 Features by PC")
ax2.legend()

# Heatmap of cluster means
ax3 = axes[1,0]
df_means = pd.DataFrame(df_processed, columns=processed_feature_names)
df_means['cluster'] = df['cluster'].values
cluster_means = df_means.groupby('cluster').mean()
sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
ax3.set_title("**Feature Means of each Cluster**")

# Empty plot for spacing
axes[1,1].axis('off')

plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# Cluster characteristics table (below)
# -------------------------------
# Numeric columns → mean
cluster_means_unstd = df.groupby('cluster')[numeric_cols].mean().round()

# Ordinal columns → mode (most common value)
cluster_modes_ordinal = df.groupby('cluster')[ordinal_cols].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# Categorical (non-ordinal) columns → mode
cluster_modes_cat = df.groupby('cluster')[categorical_cols].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# Combine all
cluster_summary = pd.concat([cluster_means_unstd, cluster_modes_ordinal, cluster_modes_cat], axis=1)

st.write("# **Features Characteristics** of each Cluster")
st.caption("Numeric features show mean values per cluster. Ordinal and Categorical features show most common value (mode) per cluster.")
st.dataframe(cluster_summary)



# -------------------------------
# Interactive 2D/3D PCA plots (pastel colors)
# -------------------------------
df_pca_plot = pd.DataFrame(df_pca, columns=['PC1','PC2','PC3'])
df_pca_plot['cluster'] = df['cluster']
hover_cols = numeric_cols + ordinal_cols
df_pca_plot = pd.concat([df_pca_plot, df[hover_cols].reset_index(drop=True)], axis=1)

# 2D PCA
fig_2d = px.scatter(df_pca_plot, x='PC1', y='PC2', color='cluster',
                    hover_data=hover_cols, color_discrete_sequence=px.colors.sequential.Viridis_r,
                    title="Interactive 2D PCA")
st.plotly_chart(fig_2d, use_container_width=True)

# 3D PCA
fig_3d = px.scatter_3d(df_pca_plot, x='PC1', y='PC2', z='PC3', color='cluster',
                       hover_data=hover_cols, color_discrete_sequence=px.colors.sequential.Magma_r,
                       title="Interactive 3D PCA")
st.plotly_chart(fig_3d, use_container_width=True)

# -------------------------------
# Statistical tests with significance
# -------------------------------
st.write("### Statistical Tests for Numeric Features")
st.info("Using ANOVA test to check if numeric feature means differ significantly across clusters.")
st.caption("If data is non-normal, Kruskal-Wallis test (non-parametric) can be used instead. We assume normality with ANOVA here for simplicity.")
numeric_results = []
for col in numeric_cols:
    groups = [df[df['cluster']==i][col] for i in df['cluster'].unique()]
    f_stat, p_val = f_oneway(*groups)
    numeric_results.append((col,p_val.round(3),
                            "Yes" if p_val<0.05 else "No",
                            ))
numeric_df = pd.DataFrame(numeric_results, columns=['Feature','ANOVA p-value',
                                                    'Are Feature Means Statistically Different from Each Cluster?'])
st.dataframe(numeric_df)

st.write("### Statistical Tests for Categorical Features")
st.info("Using Chi-square test to check if categorical features are associated with cluster membership.")
cat_results = []
for col in categorical_cols:
    contingency = pd.crosstab(df['cluster'], df[col])
    chi2, p, dof, ex = chi2_contingency(contingency)
    cat_results.append((col,p.round(3),"Yes" if p<0.05 else "No"))
cat_df = pd.DataFrame(cat_results, columns=['Feature','Chi-square p-value','Are Feature Categories Statistically Associated with Clusters?'])
st.dataframe(cat_df)

# -------------------------------
# More info on interpreting tests
# -------------------------------
st.write("## Information on Statistical Tests and their Interpretation")
info_option = st.selectbox(
    "Select an evaluation metric of interest:",
    [
        "None",
        "Optimal k",
        "Second Derivative of Inertia",
        "PCA",
        "ANOVA / Kruskal-Wallis",
        "Chi-Square Test",
        "Calinski-Harabasz Score",
        "Davies-Bouldin Score"
    ], index = 0
)
if info_option == "Optimal k":
    st.info("""The **optimal number of clusters (k)** groups your data so that each cluster is distinctly different from other clusters, yet points within a cluster are closely related.
- Too few clusters → distinct groups are combined.
- Too many clusters → clusters become fragmented and less meaningful.
- Determined using:
  - Elbow method (inertia / sum of squared distances)
  - Calinski-Harabasz Score (higher is better)
  - Davies-Bouldin Score (lower is better)
""")
elif info_option == "Second Derivative of Inertia":
    st.info("This is the **rate of change of the rate of change in inertia**. It tells us when adding more clusters stops improving cluster separation significantly. It's highest value suggests the optimal number of clusters.")
elif info_option == "PCA":
    st.info("PCA (Principal Component Analysis) reduces the dimensionality of the data for visualization "
            "while preserving most of the variance. It helps us see clusters in 2D or 3D plots.")
elif info_option == "ANOVA / Kruskal-Wallis":
    st.info("ANOVA tests whether numeric feature means are significantly different across clusters. "
            "Kruskal-Wallis is the non-parametric alternative for non-normal data. "
            "p-value < 0.05 indicates significant differences between clusters.")
elif info_option == "Chi-Square Test":
    st.info("Chi-square tests whether categorical features are associated with cluster membership. "
            "p-value < 0.05 indicates a significant association.")
elif info_option == "Calinski-Harabasz Score":
    st.info("Calinski-Harabasz Score measures the ratio of between-cluster variance to within-cluster variance. "
            "Higher scores indicate better-defined clusters.")
elif info_option == "Davies-Bouldin Score":
    st.info("Davies-Bouldin Score measures average similarity between each cluster and its most similar cluster. "
            "Lower scores indicate better cluster separation.")


##################################################################################
st.markdown("---")
st.title("Check Out Jayelle's Portfolio!")

st.markdown("""
Welcome! Here are some of my personal websites and portfolio pages where you can learn more about me and my work:
""")

# List of links
links = {
    "My Website": "https://jayelle0609.github.io/jialing",
    "Tableau Visualizations": "https://public.tableau.com/app/profile/jialingteo/vizzes",
    "GitHub Portfolio": "https://github.com/jayelle0609/Portfolio",
    "Linkedin" : "https://www.linkedin.com/in/jialingteo/",
    "Prediction App using Regression (for interview)" : "jialingpredict.streamlit.app"

}

for name, url in links.items():
    st.markdown(f"- [{name}]({url})")

st.markdown("""
---
*Feel free to reach out or explore more!*  
<span style="font-size:10px;">
[Email Me!](mailto:jayelleteo@gmail.com) | [WhatsApp Me!](https://wa.me/6580402496)
</span>
<br>
<span style="font-size:12px; color:gray;">

</span>
""", unsafe_allow_html=True)
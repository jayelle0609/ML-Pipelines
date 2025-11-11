# ==============================================
# FULL CLASSIFICATION PIPELINE (Train + Validation)
# ==============================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------------
# Classification Models
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,  IsolationForest 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


### if got na before eda ###
df.info()
df.isnull().sum()
num_cols = df.select_dtypes(include = np.number)
num_cols.fillna(num_cols.mean(), inplace=True)
cat_cols = df.select_dtypes(exclude = np.number)
cat_cols.fillna(cat_cols.mode().iloc[0], inplace=True)
# combine 
df = pd.concat([num_cols, cat_cols], axis=1)
df.info()
############################################################################# EDA Categorical Distributions, Countplot #####################################################################################
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# Define grid size (2x2)
n_rows, n_cols = 2, 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols[:n_rows * n_cols]):  # plot up to 4 categorical vars
    # Sort categories by frequency (high to low)
    order = df[col].value_counts().index

    ax = sns.countplot(x=col, data=df, order=order, ax=axes[i], palette="magma")

    axes[i].set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)

    # Add value labels above each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

# Remove unused axes if fewer than 4 categorical vars
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Global title and layout
fig.suptitle("Categorical Feature Distributions", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
############################################################################# EDA Categorical Distributions, Countplot #####################################################################################

################################################### EDA Continuous Numeric Distributions, histplot(kde = True, hue ='sub_cat') #####################################################################################
# Select numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Define grid
n_rows, n_cols = 2, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols[:n_rows * n_cols]):  # limit to 4 for 2x2 grid
    sns.histplot(data = df, x=col, kde=True, bins=20, hue = 'time', ax=axes[i], color='teal')
    axes[i].set_title(f"Distribution of {col}", fontsize=13, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Continuous Feature Distributions (Histogram + KDE)", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

################################################### EDA Continuous Numeric Distributions, histplot(kde = True) #######################################################################################

################################################### EDA Target Variable Category Pie Distributions, plt.pie() #######################################################################################
# Choose your target column
target_col = 'sex'   # replace this with your own target column name

# Count category frequencies
counts = df[target_col].value_counts()
labels = counts.index
sizes = counts.values

# Define colors (optional)
colors = sns.color_palette('pastel')[0:len(labels)]

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',      # show percentages with 1 decimal place
    startangle=90,          # rotate so largest slice starts at the top
    colors=colors, textprops={'fontsize': 20, 'color': 'black'},
    wedgeprops={'edgecolor': 'white'}  # cleaner edges
)
plt.title(f"Distribution of {target_col}", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
################################################### EDA Target Variable Category Pie Distributions, plt.pie() #######################################################################################
################################################### EDA Corr Heatmap #######################################################################################

# map categorical variables to int
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1}).astype(int)
df['time'] = df['time'].map({'Lunch': 0, 'Dinner': 1}).astype(int)
df['smoker'] =df['smoker'].map({'No': 0, 'Yes': 1}).astype(int)

# corr plot
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    mask=mask,                
    cmap='coolwarm',           
    annot=True,                
    fmt=".2f",          
    square=True,               
    annot_kws={"size": 12} 
)
plt.title("Correlation Heatmap (Lower Triangle)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

################################################### EDA Corr Heatmap #######################################################################################
################################################## Numerical rs with scatter and line plot and 95 ci ##########################################################
# Select numeric columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # flatten to iterate easily

# Loop through numerical columns
for i, col in enumerate(num_cols[:4]):  # only first 4 numeric columns to fit 2x2
    ax = axes[i]
    # Scatter + regression line
    sns.regplot(
        x=col,
        y='body_mass_g',  # example target for regression
        data=df,
        ax=ax,
        ci=95,            # 95% confidence interval
        scatter_kws={'alpha':0.6},  # transparency for scatter points
        line_kws={'color':'red'}
    )
    ax.set_title(f'{col} vs body_mass_g', fontsize=12, fontweight='bold')

# Hide empty axes if fewer than 4 numeric cols
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
################################################## Numerical rs with scatter and line plot and 95 ci #########################################################
### scatter + line + hue of a specific class type#################
num_cols = df.select_dtypes(include=np.number).columns.tolist()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

class = df['class'].unique() # select a class (penguin type 1, 2 ,3 )
colors = sns.color_palette('Set1', n_colors=len(class))

for i, col in enumerate(num_cols[:4]):
    ax = axes[i]
    for clazz, color in zip(class, colors):
        class_data = df[df['class'] == clazz]
        sns.scatterplot(x=col, y='body_mass_g', data=class_data, ax=ax, color=color, label=clazz, alpha=0.6) # replace with one of the cols of interest
        sns.regplot(x=col, y='body_mass_g', data=class_data, ax=ax, scatter=False, line_kws={'color': color})
    ax.set_title(f'{col} vs body_mass_g', fontsize=12, fontweight='bold')

# Hide empty axes if fewer than 4 numeric columns
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
################################################## Box plot to detect outliers in num cols #########################################################
# Select numeric columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Determine layout (2x2 grid here as example)
n_cols = 2
n_rows = int(np.ceil(len(num_cols)/n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows*4))
axes = axes.flatten()  # flatten for easy iteration

# Loop through numeric columns
for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(x=df[col], ax=ax, color='skyblue')
    ax.set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')

# Hide empty axes if any
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
ordinal_cols = []
ordered_categories = []
################################################## Box plot to detect outliers in num cols #########################################################


# Usually remove outliers for numerical data only. Cat data - no rly such thing as outliers 
# -------------------------------
# Outlier Removal with Isolation Forest (VIZ PART)
# -------------------------------
# We drop 'tip' as we don’t want to remove outliers based on the target, only on features
numeric_cols = df.select_dtypes(include=np.number).columns.drop('target', errors='ignore')
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(df[numeric_cols])
mask = iso.predict(df[numeric_cols]) != -1
df['is_outlier'] = ~mask  # True = outlier, False = inlier
####################################################### VIZ 0 : out lier bar plots with no annotate############################################
df['is_outlier'].value_counts().plot(kind='bar') 
 ############################################ VIZ 0 : out lier bar plots with annotate############################################
ax = df['is_outlier'].value_counts().rename({True : "Outlier", False : "Not Outliers"}).plot(kind='bar', color=['blue', 'orange'])
# Annotate values directly on the bars
for i, v in enumerate(df['is_outlier'].value_counts()):
    ax.text(i, v + 1, str(v), ha='center', fontsize=12, color='black')
    ax.tick_params(axis = 'x', rotation = 45)
# Display the plot
plt.title('Outlier Distribution')
plt.xlabel('Outlier (True/False)')
plt.ylabel('Count')
plt.show()
#################################################################################################
# Viz 1 - outlier
plt.figure(figsize=(8,6))
plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]],
            c=df['is_outlier'], cmap='coolwarm', edgecolor='k')
plt.xlabel(numeric_cols[0])
plt.ylabel(numeric_cols[1])
plt.title('Outliers detected by Isolation Forest')
plt.show()
#############################################################################
## Viz 2 - outlier 
sns.pairplot(df, vars=numeric_cols, hue='is_outlier', palette={False:'blue', True:'red'}, corner = True) # masks repeating graphs
#############################################################################
# Viz 3 - outlier
df['anomaly_score'] = iso.decision_function(df[numeric_cols]) # shows u outlier anomaly score, higher = worser anomaly
plt.figure(figsize=(8,4))
plt.hist(df['anomaly_score'], bins=30, color='skyblue', edgecolor='k')
plt.axvline(x=0, color='red', linestyle='--', label='Threshold')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Distribution of Isolation Forest Scores')
plt.legend()
plt.show()
#############################################################################
# Viz 4 - PCA outliers + labels of anomaly score 
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[numeric_cols])
# Store the first 2 principal components
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]
# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Plotting
scatter = ax.scatter(df['PC1'], df['PC2'], c=df['is_outlier'], cmap='coolwarm', edgecolor='k')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Isolation Forest Outliers in PCA Space')
# Label outliers with their anomaly scores
outliers = df[df['is_outlier']]  # Filtering the outliers
for _, row in outliers.iterrows():
    ax.text(
        row['PC1'], row['PC2']+0.2, f"{row['anomaly_score']:.2f}", fontsize=10, color='red', ha='center', va='center'
    )
# Add color legend for outliers
ax.legend(*scatter.legend_elements(), title="Outliers")
# Adding annotation text below the plot (on the canvas, not on the axes)
annotation_text = (
    "Low anomaly scores (closer to -1) indicate outliers or anomalies.\n"
    "High anomaly scores (closer to 1) indicate inliers or normal points"
)
# Add the annotation text **below the entire plot**
fig.text(0.5, -0.04, annotation_text, ha='center', fontsize=10, color='black', va='center', wrap=True)
# Show the plot
plt.tight_layout()  # Ensures everything fits nicely
plt.show()
#############################################################################
# Viz 5 - 3D view of outliers
# Run 3D PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[numeric_cols])
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]
df['PC3'] = pca_result[:, 2]
# 3D scatter plot
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',color='is_outlier', color_discrete_map={False: 'blue', True: 'red'},hover_data=df.columns,  # shows anomaly score + others cols info on hover
    title='Isolation Forest Outliers in 3D PCA space')
fig.show()

# -------------------------------------
# Outlier Removal now! (REMOVAL PART)
# -------------------------------------
# PRINTED EARLIER
# numeric_cols = df.select_dtypes(include=np.number).columns.drop('target', errors='ignore')
# iso = IsolationForest(contamination=0.05, random_state=42)
# iso.fit(df[numeric_cols])
# mask = iso.predict(df[numeric_cols]) != -1
# df['is_outlier'] = ~mask  # True = outlier, False = inlier
df = df[mask].reset_index(drop=True)
##########################################################################
df = df.drop(columns= ['is_outlier', 'anomaly_score', 'PC1', 'PC2', 'PC3'])
X = df.drop(columns='y') # drop target col
y = df['target']
le = LabelEncoder()
y = le.fit_transform(df['y'])
# Check the mapping
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Select numeric and categorical columns
# -------------------------------
# Note not to feed y into pipeline. 
# y should not be scaled 
numeric_cols = X.select_dtypes(include=np.number).columns.tolist() # or X
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist() # or X
 
ordinal_cols = ['education']
education_order = ['low', 'medium', 'high']

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

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
    ('ord', ordinal_pipeline, ordinal_cols)
])

# -------------------------------
# Model Selection
# -------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
}

# -------------------------------
# Train/Validation Split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    df, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Optional Sampling
sample_frac = 0.1
X_sample = X_train.sample(frac=sample_frac, random_state=42)
y_sample = y_train.loc[X_sample.index]

# -------------------------------
# Initial CV (for initial model comparison, use X_train/y_train or X_sample/y_sample only.)
# -------------------------------
# X_val and y_val are kept aside for final validation after hyperparameter tuning

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
#
    acc_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_weighted') # 'f1_weighted' for imbalanced datasets

    #acc_scores = cross_val_score(pipeline, X_sample, y_sample, cv=kf, scoring='accuracy')
    #f1_scores = cross_val_score(pipeline, X_sample, y_sample, cv=kf, scoring='f1')   

    # Optional: ROC-AUC (only for models with predict_proba)
    if hasattr(model, "predict_proba"):  # <-- check the model
        y_proba_cv = cross_val_predict(pipeline, X_train, y_train, cv=skf, method='predict_proba')
        n_classes = len(np.unique(y_train))

        if n_classes == 2:  # Binary classification
            roc_auc = roc_auc_score(y_train, y_proba_cv[:, 1])
        else:  # Multiclass classification
            roc_auc = roc_auc_score(y_train, y_proba_cv, multi_class='ovr', average='weighted')

        print(f"{name} Cross-validated ROC-AUC: {roc_auc:.3f}")
    else:
        roc_auc = np.nan

    cv_results.append({
        'Model': name,
        'Accuracy Mean': np.mean(acc_scores),
        'F1 Mean': np.mean(f1_scores),
        'Acc Std': np.std(acc_scores),
        'F1 Std': np.std(f1_scores),
        'ROC-AUC': np.mean(roc_auc)
    })

cv_df = pd.DataFrame(cv_results).sort_values(by='F1 Mean', ascending=False)
print(cv_df)

    # Store results
    cv_results.append({
        'Model': name,
        'Accuracy Mean': acc_scores.mean(),
        'F1 Mean': f1_scores.mean(),
        'Acc Std': acc_scores.std(),
        'F1 Std': f1_scores.std(),
        'ROC-AUC': roc_auc
    })

cv_df = pd.DataFrame(cv_results).sort_values(by='F1 Mean', ascending=False)
print("\nInitial CV Results:\n", cv_df)

best_model_name = cv_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nBest model selected: {best_model_name}")

# -------------------------------
# Plotting best model metrics vs other models 
# -------------------------------
metrics = ['F1 Mean', 'Accuracy Mean', 'ROC-AUC','F1 Std', 'Acc Std']
cv_df_sorted = cv_df.sort_values(by='F1 Mean', ascending=False)
models_list = cv_df_sorted['Model'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(
        x='Model', 
        y=metric, 
        data=cv_df_sorted, 
        ax=ax, 
        palette='magma'
    )
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(models_list, rotation=45, ha='right')
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11)

# Use the last subplot for custom text
if len(metrics) < len(axes):
    ax_text = axes[-1]
    ax_text.axis('off')  # turn off axes
    summary_text = (
        "Insights:\n"
        "• Higher F1 Mean, Accuracy, and ROC-AUC = better model performance.\n"
        "• Lower F1 Std and Accuracy Std = more stable and consistent model predictions.\n"
        "• Best Model : ______ \n"
    )
    ax_text.text(0.5, 0.5, summary_text,
                 ha='center', va='center', fontsize=12, fontweight='bold', wrap=True)

plt.suptitle("Model Comparison Across Metrics", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# -------------------------------
# Select best model
# -------------------------------
best_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss'
)

# -------------------------------
# Build pipeline
# -------------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

pipeline.fit(X_train, y_train)

# -------------------------------
# Save CV Predictions on baseline model FOR TEST.CSV
# -------------------------------
test = pd.read_csv('test.csv')  # Load your test set here

y_test_pred = pipeline.predict(test)

# Optional probabilities for ROC-AUC
if hasattr(best_model, "predict_proba"):
    y_test_proba = pipeline.predict_proba(test)[:, 1]

# Save results
pred_test_df = pd.DataFrame({'Predicted': y_test_pred, 'Probability': y_test_proba if 'y_test_proba' in locals() else np.nan})
pred_test_df.to_csv('test_predictions.csv', index=False)
print("✅ Test predictions saved to 'test_predictions.csv'")


# -------------------------------
# Hyperparameter Tuning
# -------------------------------
# Example: adjust depending on your model
# ---------------------------------------
best_model = ...
### Logistic Regression ###
param_dist_lr = {
    'model__C': np.logspace(-3, 2, 6),
    'model__penalty': ['l2'],
    'model__solver': ['lbfgs', 'saga'],
    'model__max_iter': [200, 500]
}

### Random Forest Classifier ###
param_dist_rf = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [None, 5, 10, 15],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None],
    'model__bootstrap': [True, False]
}

### Gradient Boosting Classifier ###
param_dist_gb = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [2, 3, 4, 5],
    'model__subsample': [0.6, 0.8, 1.0]
}

### XGBoost Classifier ###
param_dist_xgb = {
    'model__n_estimators': np.arange(200, 800, 100),
    'model__max_depth': np.arange(3, 10),
    'model__learning_rate': np.linspace(0.01, 0.3, 5),
    'model__subsample': np.linspace(0.6, 1.0, 5),
    'model__colsample_bytree': np.linspace(0.6, 1.0, 5),
    'model__gamma': [0, 1, 2],
    'model__reg_alpha': [0, 0.5, 1],
    'model__reg_lambda': [1, 2, 3]
}

### LightGBM Classifier ###
param_dist_lgbm = {
    'model__n_estimators': [200, 400, 600],
    'model__num_leaves': [31, 50, 100],
    'model__max_depth': [-1, 5, 10],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.6, 0.8, 1.0]
}

### CatBoost Classifier ###
param_dist_cat = {
    'model__iterations': [200, 500, 800],
    'model__depth': [4, 6, 8, 10],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__l2_leaf_reg': [1, 3, 5]
}


# Choose parameter grid to match the model
param_dist = param_dist_xgb  # ← change this based on model you are using

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1_weighted', # 'f1_weighted' for imbalanced datasets
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# -------------------------------
# Run Hyperparameter Tuning
# -------------------------------
random_search.fit(X_train, y_train)
print("✅ Hyperparameter tuning completed.")
print("Best Parameters:", random_search.best_params_)
print("Best Estimator:", random_search.best_estimator_) # prints full pipeline with best params
print("Best CV F1 Weighted Score :", random_search.best_score_) 

# Validation
y_pred_val = random_search.predict(X_val)
y_pred_val_proba = random_search.predict_proba(X_val)[:, 1] if hasattr(random_search.best_estimator_.named_steps['model'], "predict_proba") else None
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val)) #
print(classification_report(y_val, y_pred_val))
pred_val_df = pd.DataFrame({'Actual': y_val, 'Val-Predicted': y_pred_val, "Val-Predicted_Prob": y_pred_val_proba})


# -------------------------------
# If you have a test set to predict from
# -------------------------------
# -------------------------------
# Test predictions after hyperparameter tuning
# -------------------------------
try:
    # Predict on test set
    y_pred_test = random_search.predict(test)  

    # Optional probabilities for ROC-AUC (binary classification)
    if hasattr(random_search.best_estimator_.named_steps['model'], "predict_proba"):
        y_test_proba = random_search.predict_proba(test)[:, 1]

    # Save predictions
    pred_test_df = pd.DataFrame({'Predicted': y_pred_test})
    
    # Include probabilities if available
    if 'y_test_proba' in locals():
        pred_test_df['Predicted_Prob'] = y_test_proba

    pred_test_df.to_csv('test_predictions.csv', index=False)
    print("✅ Test predictions saved to 'test_predictions.csv'")

except Exception as e:
    print(f"❌ Error during test predictions: {e}")



# -------------------------------
# Visualization Function (binary class)
# -------------------------------
def plot_classification_results_val(y_val, y_pred_val, y_pred_val_proba=None):
    plt.figure(figsize=(14, 5))

    # 1️⃣ Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_val, y_pred_val)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # 2️⃣ ROC Curve (if probabilities provided)
    plt.subplot(1, 3, 2)
    if y_pred_val_proba is not None:
        fpr, tpr, _ = roc_curve(y_val, y_pred_val_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
    else:
        plt.text(0.5, 0.5, 'ROC unavailable', ha='center', va='center')

    # 3️⃣ Predicted Class Distribution
    plt.subplot(1, 3, 3)
    sns.countplot(x=y_pred_val, palette='pastel')
    plt.title("Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

# -------------------------------
# Plot Validation Results
# -------------------------------
plot_classification_results_val(y_val, y_pred_val, y_pred_val_proba)


# -------------------------------
# Visualization Function (multi-class)
# -------------------------------
def plot_classification_results_val_multi(y_val, y_pred_val, y_pred_val_proba=None):
    plt.figure(figsize=(14, 5))  # slightly narrower for 2 plots

    # 1️⃣ Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_val, y_pred_val, labels=np.unique(y_val))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # 2️⃣ Predicted Class Distribution
    plt.subplot(1, 2, 2)
    ax = sns.countplot(x=y_pred_val, palette='pastel', order=np.unique(y_val))
    plt.title("Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")

    # Add counts on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                    xy=(p.get_x() + p.get_width() / 2, height), 
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Call the function
plot_classification_results_val_multi(y_val, y_pred_val, y_pred_val_proba)




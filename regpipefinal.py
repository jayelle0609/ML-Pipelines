
# -------------------------------
# Full Numerical Pipeline with Outlier Handling and VIF Check
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

######### if got na before eda #######
df.info()
df.isnull().sum()
num_cols = df.select_dtypes(include = np.number)
num_cols.fillna(num_cols.mean(), inplace=True)
cat_cols = df.select_dtypes(exclude = np.number)
cat_cols.fillna(cat_cols.mode().iloc[0], inplace=True)
# combine 
df = pd.concat([num_cols, cat_cols], axis=1)
df.info()
df1.merge(df2, how = "inner", on="colname")
df.melt(id_vars = "coltokeepasidx", value_vars = ['colstomelt'], value_name=['newcolname'])
############################################################################# EDA Categorical Distributions, Countplot #####################################################################################
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# Define grid size (2x2)
n_rows, n_cols = 2, 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols[:n_rows * n_cols]):  # plot up to 4 categorical vars
    # Sort categories by frequency (high to low)
    order = df[col].value_counts().index

    ax = sns.countplot(x=col, data=df, order=order, ax=axes[i], palette="viridis")

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

survive = df['survived'].unique()  # select unique classes
colors = sns.color_palette('Set1', n_colors=len(survive))

for i, col in enumerate(num_cols[:4]):  # first 4 numeric columns for 2x2 grid
    ax = axes[i]
    for clazz, color in zip(survive, colors):
        class_data = df[df['survived'] == clazz]
        sns.scatterplot(x=col, y='age', data=class_data, ax=ax, color=color, label=str(clazz), alpha=0.6)
        sns.regplot(x=col, y='age', data=class_data, ax=ax, scatter=False, line_kws={'color': color})
    ax.set_title(f'{col} vs age', fontsize=12, fontweight='bold')

# Hide unused axes if fewer than 4 numeric columns
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

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
ordinal_cols = []
ordered_categories = []



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
# scale first before fitting into PCA as PCA is affected by scale 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

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
pca_result = pca.fit_transform(X_scaled)
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]
df['PC3'] = pca_result[:, 2]
# 3D scatter plot
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',color='is_outlier', color_discrete_map={False: 'blue', True: 'red'},hover_data=df.columns,  # shows anomaly score + others cols info on hover
    title='Isolation Forest Outliers in 3D PCA space')
fig.update_layout(width=700, height=600)
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
# Numeric preprocessing
# -------------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # or median
    ('scaler', StandardScaler())
])

# -------------------------------
# Categorical preprocessing
# -------------------------------
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # mode
    ('onehot', OneHotEncoder(handle_unknown='ignore', 
                             drop = "first", 
                             sparse_output = False)) # avoid dumy trap
])

# -------------------------------
# Ordinal preprocessing
# -------------------------------
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[education_order]))
])

# -------------------------------
# Full preprocessing
# -------------------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
    ('ord', ordinal_pipeline, ordinal_cols)
])

# -------------------------------
# Model Selection
# -------------------------------

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42), 
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='rmse'),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0)
    }


# -------------------------------
# Initial CV on small sample
# -------------------------------
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# X = train.drop(columns = 'y')   # dont do this if u dropped earlier
# y = train['y']          

# No need for val, as cv will internally settle validation set (x no. of folds)
# X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # this was done earlier


sample_frac = 0.1  # 10% sample for initial testing
X_sample = X.sample(frac=sample_frac, random_state=42)
y_sample = y.loc[X_sample.index]

kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    print(f"Training model: {name}")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    # CV RMSE
    mse_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1) # or X_sample, y_sample
    rmse_scores = np.sqrt(-mse_scores)
    # CV MAE
    mae_scores = cross_val_score(pipeline, X_train, y_train, cv = kf, scoring='neg_mean_absolute_error', n_jobs=-1)
    # CV R²
    r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
    
    # Append results for this model
    cv_results.append({
        'Model': name,
        'R2 Mean': r2_scores.mean(),
        'RMSE Mean': rmse_scores.mean(),
        'R2 Std': r2_scores.std(),
        'RMSE Std': rmse_scores.std(),
    })
# In machine learning, we generally try to maximize the scoring function. For example, when using R² (coefficient of determination), higher values are better, so R² is already a maximizing metric.
# However, loss functions like MAE or MSE are minimizing metrics (lower values mean better performance).
# To handle this in a consistent way, scikit-learn negates the values of minimizing metrics (like MAE, MSE) when they are used as the scoring function. This converts the problem of minimizing error into a maximization problem.
# So, scikit-learn multiplies the MAE values by -1 (i.e., negative MAE) to make the optimization consistent with the maximization paradigm.

# Create DataFrame and sort
cv_df = pd.DataFrame(cv_results).sort_values(by='R2 Mean', ascending = False)
print("Initial CV Results on Sample:\n", cv_df)

# Select best model based on RMSE and R2
best_model_name = cv_df.iloc[0]['Model']
best_model = models[best_model_name]

print('-' *100)
print(f"Best model selected: {best_model_name}")

################# PLOT RESULTS FOR MODEL SHORTLISTING #########################
plot_df = cv_df.copy()
# 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10,6))
axes = axes.flatten()  # flatten to iterate easily
def plot_sorted_bar(ax, y_col, palette, ascending=True):
    # Sort the dataframe by the metric
    sorted_df = plot_df.sort_values(by=y_col, ascending=ascending)
    sns.barplot(x='Model', y=y_col, data=sorted_df, palette=palette, ax=ax)
    ax.set_title(y_col)
    ax.tick_params(axis='x', rotation=45)
    # Annotate each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
# R² Mean (highest first)
plot_sorted_bar(axes[0], 'R2 Mean', 'Greens_r', ascending=False)
# RMSE Mean (lowest first)
plot_sorted_bar(axes[1], 'RMSE Mean', 'Reds_r', ascending=True)
# MAE Mean (lowest first)
plot_sorted_bar(axes[2], 'R2 Std', 'Blues_r', ascending=True)
plot_sorted_bar(axes[3], 'RMSE Std', 'Blues_r', ascending=True)

# Large title for whole figure
fig.suptitle("Cross-Validation Metrics Comparison Across Models", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.text(0.5, -0.1, 
         "A lower standard deviation indicates that the model's performance is more consistent \n"
         "across the different cross-validation folds, and suggests that the model is stable and reliable.",
         ha='center', va='center', fontsize=12, fontweight='normal', wrap=True)
plt.show()
################# PLOT RESULTS FOR MODEL SHORTLISTING #########################

# Might not be needed, only for viz
val_metrics_bestmodel_df = pd.DataFrame({'CV Mean Scores' : ['R2', 'RMSE' , 'MAE'], 'Values' : [r2_scores.mean(), rmse_scores.mean(),  mae_scores.mean()]})

# -------------------------------
# Fit best model on whole data set
# -------------------------------
best_model = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)])
        
pipeline.fit(X_train, y_train)

# -------------------------------
# Predictions on test set with baseline model (never do CV on test set pls!!!) Dun let model see the data or it will train on it
# -------------------------------
try: 
    y_pred = pipeline.predict(X_test)
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    pred_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test , "Residuals" : y_test - pd.Series(y_pred, index=y_test.index)}) # or y_test - y_pred will work too. but to be safe
    metrics_baseline_bestmodel_df = pd.DataFrame({"Test Scores (Before Hyperparameter Tuning)" : ["R2", "MAE","RMSE"], "Values" : [r2, mae, rmse]})
    metrics_baseline_bestmodel_df['Values'] = metrics_baseline_bestmodel_df['Values'].apply(lambda x: round(x, 2)) # round to 2sf
    print(pred_df)
    print(metrics_baseline_bestmodel_df)
    # Save to CSV
    pred_df.to_csv('predictions_baseline_model.csv', index=False)
    print("✅ Predictions saved to 'predictions_baseline_model.csv'")

except Exception as e:
    print(f"An error occurred while training {name}: {e}")
  
############################################################### Plotting best model : Predicted vs Actual, Hued by train and validation, with 45 degree line################################################
# Predict on train and test
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Create a combined DataFrame for plotting
pred_plot_df = pd.DataFrame({
    'Actual': list(y_train) + list(y_test),
    'Predicted': list(y_train_pred) + list(y_test_pred),
    'Set': ['Train']*len(y_train) + ['Validation']*len(y_test)
})

# Plotting Predicted vs Actual, hued by trg or validation, with 45 degree line 
plt.figure(figsize=(8,8))

# Scatter plot with hue for Train vs Validation
sns.scatterplot(data=pred_plot_df, x='Predicted', y='Actual', hue='Set', alpha=0.6, palette=['blue', 'red'])

# 45-degree reference line (perfect prediction)
min_val = min(pred_plot_df['Actual'].min(), pred_plot_df['Predicted'].min())
max_val = max(pred_plot_df['Actual'].max(), pred_plot_df['Predicted'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Prediction')

plt.title('Predicted vs Actual - Train vs Validation', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
############################################################### Plotting best model : Predicted vs Actual, Hued by train and validation, with 45 degree line################################################
# -------------------------------
# Hyperparameter Tuning with RandomSearchCV
# -------------------------------

#### This is for XGBoost, LightGBM, or Catboost ####
param_dist = {
    'model__n_estimators': np.arange(200, 1000, 100),
    'model__max_depth': np.arange(3, 10),
    'model__learning_rate': np.linspace(0.01, 0.3, 5),
    'model__subsample': np.linspace(0.6, 1.0, 5),
    'model__colsample_bytree': np.linspace(0.6, 1.0, 5),
    'model__gamma': [0, 1, 2],
    'model__reg_alpha': [0, 0.5, 1],
    'model__reg_lambda': [1, 2, 3]
}
#### This is for Gradient Boosting Regressor ####
param_dist = {
    'model__n_estimators': [100, 200, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [2, 3, 4, 5],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__max_features': ['sqrt', 'log2', None],
    'model__loss': ['squared_error', 'absolute_error', 'huber']
}

#### This is for Random Forest Regressor ####
param_dist = {
    'model__n_estimators': [100, 300, 500, 800],
    'model__max_depth': [None, 5, 10, 15],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None],
    'model__bootstrap': [True, False]
}

#### This is for Linear, Ridge, Lasso Regressor ####
from scipy.stats import loguniform
param_dist = {
    'model__alpha': np.logspace(-3, 2, 6),  --> test 6 values only boo # regularization strength #  loguniform(1e-4, 1e4) --> test from __ to ___
    'model__fit_intercept': [True, False],
    'model__copy_X': [True]
}


random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,  # number of random combos to try
    scoring='r2', # 'r2' # only can use 1 metric 
    cv=3, # CROSS VALIDATEEEEEEEE
    n_jobs=-1, # run faster!
    random_state=42,
    verbose=2
)

# -------------------------------
# Hyperparameter Tuning Best Params
# -------------------------------
# Fit the best params into best model
try:
    random_search.fit(X_train, y_train)
    print("Hyperparameter tuning completed on training set.")
    print("Best Parameters:", random_search.best_params_)
    print("Best Estimator:", random_search.best_estimator_) # prints full pipeline
    print("Best CV R²:", random_search.best_score_) # CValidation scores
    
    # Validation predictions
    y_pred_val = random_search.predict(X_val) # this should be X_test if no separate test.csv and y_pred_test
    mse = mean_squared_error(y_val, y_pred_val) #y_test, y_pred_test
    mae = mean_absolute_error(y_val, y_pred_val) #y_test, y_pred_test
    r2 = r2_score(y_val, y_pred_val) #y_test, y_pred_test
    
    print(f"Validation RMSE: {np.sqrt(mse):,.2f}")
    print(f"Validation MAE: {mae:,.2f}")
    print(f"Validation R2 Score: {r2:.2f}")
    
    # Apply to test set (if u have a separate test.csv)
    y_pred_test = random_search.predict(X_test)
    print("Test predictions:", y_pred_test)
    pred_test_df = pd.DataFrame({'Predicted': y_pred_test})
    pred_test_df.to_csv('test_predictions.csv', index=False)
    print("✅ Test predictions saved to 'test_predictions.csv'")
    

except Exception as e:
    print(f"An error occurred: {e}")


############## to check if default model or optuna best parms is better##########################################################
############## to check if default model or optuna best parms is better##########################################################
! pip install optuna
# ---------------------------
# Full Pipeline: Baseline vs Optuna for Random Forest
# ---------------------------
# 2️⃣ Load dataset
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3️⃣ Baseline Random Forest
rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)

r2_default = r2_score(y_test, y_pred_default)
rmse_default = np.sqrt(mean_squared_error(y_test, y_pred_default))
print(f"Baseline RF: R² = {r2_default:.3f}, RMSE = {rmse_default:.3f}")

# 4️⃣ Optuna objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)  # maximize R²

# 5️⃣ Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# 6️⃣ Fit best Optuna model
best_params = study.best_params
rf_optuna = RandomForestRegressor(**best_params, random_state=42)
rf_optuna.fit(X_train, y_train)
y_pred_optuna = rf_optuna.predict(X_test)

r2_optuna = r2_score(y_test, y_pred_optuna)
rmse_optuna = np.sqrt(mean_squared_error(y_test, y_pred_optuna))
print(f"Optuna RF: R² = {r2_optuna:.3f}, RMSE = {rmse_optuna:.3f}")

# 7️⃣ Visualize baseline vs Optuna
models = ['Baseline', 'Optuna']
r2_scores = [r2_default, r2_optuna]
rmse_scores = [rmse_default, rmse_optuna]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# R² comparison
bars1 = ax[0].bar(models, r2_scores, color=['skyblue', 'salmon'])
ax[0].set_title("R² Comparison")
ax[0].set_ylim(0, 1)
ax[0].set_ylabel("R²")

# Label each R² value
for bar in bars1:
    height = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.3f}", ha='center', va='bottom')

# RMSE comparison
bars2 = ax[1].bar(models, rmse_scores, color=['skyblue', 'salmon'])
ax[1].set_title("RMSE Comparison")
ax[1].set_ylabel("RMSE")

# Label each RMSE value
for bar in bars2:
    height = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.3f}", ha='center', va='bottom')

plt.suptitle("Baseline vs Optuna Hyperparameter Tuning")
plt.show()

# print best params
print("Best trial and parameters found by Optuna:")
trial = study.best_trial
print("  Value (R²):", trial.value)
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 8️⃣ Optional: Optuna optimization history
try:
    import optuna.visualization as vis
    fig_optuna = vis.plot_optimization_history(study)
    fig_optuna.show()
except:
    print("Plotly not installed, skipping Optuna visualization")

###### Plot learning curve between baseline and best params model (for train and valid set) #############################
###### Plot learning curve between baseline and best params model (for train and valid set) #############################
# Function to plot learning curve
def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring='r2'):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label="Training score")
    plt.plot(train_sizes, val_mean, 'o-', color='red', label="Validation score")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.1, color='red')
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Baseline model
plot_learning_curve(rf_default, X_train, y_train, title="Learning Curve - Baseline RF", scoring='r2')

# Optuna-tuned model
plot_learning_curve(rf_optuna, X_train, y_train, title="Learning Curve - Optuna RF", scoring='r2')

############## to check if default model or optuna best parms is better##########################################################
############## to check if default model or optuna best parms is better##########################################################
# -------------------------
# ✅ Done: Default vs optuna comparison
# -------------------------

# -------------------------------
# Show plots of pred vs actual
# -------------------------------

def plot_regression_results(y_val, y_pred_val):
    """
    Visualize regression model performance for validation set after hyperparameter tuning.
    
    Creates:
    1. Actual vs Predicted scatterplot
    2. Residuals vs Actual values (Actual - Predict = Residuals)
    3. Residual distribution (histogram + KDE)
    4. Line plot comparison (Actual vs Predicted sequence)
    """
    residuals = y_val - y_pred_val #y_test, y_pred_test
    # residuals = y_test - y_pred_test
    plt.figure(figsize=(14, 12))

    # 1️⃣ Actual vs Predicted
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=y_val, y=y_pred_val, color='royalblue', alpha=0.7) #x = y_test, y = y_pred_test
    # sns.scatterplot(x=y_test, y=y_pred_test, color='royalblue', alpha=0.7)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2) 
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual (y_val)") 
    # plt.xlabel("Actual (y_test)")
    plt.ylabel("Predicted (y_pred_val)") # y_pred_test
    # plt.ylabel("Predicted (y_pred_test)")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2️⃣ Residuals vs Actual
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=y_val, y=residuals, color='darkorange', alpha=0.7)
    # sns.scatterplot(x=y_test, y=residuals, color='darkorange', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title("Residuals vs Actual Values")
    plt.xlabel("Actual (y_val)")
    # plt.xlabel("Actual (y_test)")
    plt.ylabel("Residuals")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3️⃣ Residual Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, kde=True, color='purple', bins=20)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    # 4️⃣ Line Plot (Sequence Comparison)
    plt.subplot(2, 2, 4)
    plt.plot(y_val.values, label='Actual', color='black', linewidth=2)
    # plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
    plt.plot(y_pred_val, label='Predicted', color='dodgerblue', linestyle='--')
    # plt.plot(y_pred_test, label='Predicted', color='dodgerblue', linestyle='--')
    plt.title("Actual vs Predicted (Sequence View)")
    plt.xlabel("Observation Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

###################### Viewing coefficients of linear model ##############################################
best_model = pipeline.named_steps['model']
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lasso_model.coef_
})
coef_df

###################### Plotting coefficients of linear model and viewing with intensity on bar ##############################################


# 1️⃣ Absolute value for color intensity
coef_df['AbsCoeff'] = coef_df['Coefficient'].abs()

# 2️⃣ Sort by absolute coefficient magnitude (most impactful first)
coef_df = coef_df.sort_values(by='AbsCoeff', ascending=False)

# 3️⃣ Normalize for color mapping
norm = plt.Normalize(coef_df['Coefficient'].min(), coef_df['Coefficient'].max())
colors = plt.cm.coolwarm(norm(coef_df['Coefficient'].values))

# 4️⃣ Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')

# 5️⃣ Annotate each bar with its coefficient value
for bar in bars:
    plt.text(
        bar.get_width(),                           # x-position (end of bar)
        bar.get_y() + bar.get_height() / 2,        # y-position (center of bar)
        f"{bar.get_width():.3f}",                  # formatted value
        ha='left' if bar.get_width() > 0 else 'right',
        va='center',
        fontsize=9,
        color='black',
        fontweight='bold'
    )

# 6️⃣ Styling
plt.title("Feature Coefficients (Lasso Regression)", fontsize=14, fontweight='bold')
plt.xlabel("Coefficient Value", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Rotate labels (use horizontal plot for long names)
plt.yticks(fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()

############################## Feature importances of tree models and viewing intensity on bar with annotation on bar##################################################

# Example assumes you’ve already trained your model
# (works for RandomForest, XGB, LGBM, CatBoost, etc.)
best_model = RandomForestRegressor(random_state=42)
pipeline.fit(X_train, y_train)

# Get feature importances
importances = best_model.feature_importances_
feature_names = X_train.columns

# Put into dataframe
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Normalize importance for color intensity mapping (0–1)
importances_df['NormImportance'] = importances_df['Importance'] / importances_df['Importance'].max()

# Choose a colormap — viridis is nice and perceptually uniform
colors = sns.color_palette("viridis", as_cmap=True)(importances_df['NormImportance'])

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(
    importances_df['Feature'],
    importances_df['Importance'],
    color=colors,
    edgecolor='black'
)

# Annotate values on bars (optional)
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{bar.get_height():.3f}",
        ha='center', va='bottom', fontsize=9
    )

# Rotate labels and improve layout
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Feature Importance (Colored by Intensity)', fontsize=14, fontweight='bold')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()


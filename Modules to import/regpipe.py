
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

# convert bool cols to numeric
# Map boolean to 0/1
df['smoker'].value_counts() # check only 2 values to map
data['is_male'] = data['sex'].map({Male: 1, Female: 0})
data['smoker'] = data['sex'].map({Yes: 1, No: 0})

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
# Viz 1 - outlier
plt.figure(figsize=(8,6))
plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]],
            c=df['is_outlier'], cmap='coolwarm', edgecolor='k')
plt.xlabel(numeric_cols[0])
plt.ylabel(numeric_cols[1])
plt.title('Outliers detected by Isolation Forest')
plt.show()
## Viz 2 - outlier 
sns.pairplot(df, vars=numeric_cols, hue='is_outlier', palette={False:'blue', True:'red'})
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
# Viz 4 - PCA outliers + labels of anomaly score 
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[numeric_cols])
# Store the first 2 principal components
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]
plt.figure(figsize=(8,6))
plt.scatter(df['PC1'], df['PC2'], c=df['is_outlier'], cmap='coolwarm', edgecolor='k')
# Get anomaly scores from Isolation Forest. # Labellig anomaly score on diag
outliers = df[df['is_outlier']]
for _, row in outliers.iterrows():
    plt.text(row['PC1'], row['PC2'], f"{row['anomaly_score']:.2f}", fontsize=8, color='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Isolation Forest Outliers in PCA space')
plt.show()
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
df_clean = df[mask].reset_index(drop=True)
X = df_clean.drop(columns='y') # drop target col
y = df_clean['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Select numeric and categorical columns
# -------------------------------
# Note not to feed y into pipeline. 
# y should not be scaled 
numeric_cols = df.select_dtypes(include=np.number).columns.tolist() # or X
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist() # or X
 
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
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop = "first", sparse_output = False))     # One-hot encode
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
    r2_scores = cross_val_score(pipeline, X_sample, y_sample, cv=kf, scoring='r2', n_jobs=-1)
    
    # Append results for this model
    cv_results.append({
        'Model': name,
        'RMSE Mean': rmse_scores.mean(),
        'R2 Mean': r2_scores.mean(),
        'MAE Mean' : mae_scores.mean()
        'RMSE Std': rmse_scores.std(),
        'R2 Std': r2_scores.std(),
        'MAE Std' : mae_scores.std()
    })

# Create DataFrame and sort
cv_df = pd.DataFrame(cv_results).sort_values(by='RMSE Mean')
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
plot_sorted_bar(axes[2], 'MAE Mean', 'Blues_r', ascending=True)
# Last subplot empty or optional text
axes[3].axis('off')
axes[3].text(0.5, 0.5, 'Type Analysis Summary Here', ha='center', va='center', fontsize=12, fontweight='bold')
# Large title for whole figure
fig.suptitle("Cross-Validation Metrics Comparison Across Models", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
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
param_dist = {
    'model__alpha': np.logspace(-3, 2, 6),   # regularization strength
    'model__fit_intercept': [True, False],
    'model__copy_X': [True]
}


random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,  # number of random combos to try
    scoring='neg_root_mean_squared_error', # 'r2' # only can use 1 metric 
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

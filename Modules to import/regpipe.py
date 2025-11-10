
# -------------------------------
# Full Numerical Pipeline with Outlier Handling and VIF Check
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
ordinal_cols = []
ordered_categories = []


# -------------------------------
# Custom Transformer: Outlier Removal
# -------------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Remove outliers using Z-score or IQR.
    """
    def __init__(self, method='zscore', threshold=3):
        self.method = method
        self.threshold = threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.method == 'zscore':
            from scipy.stats import zscore
            z_scores = np.abs(zscore(X))
            return X[(z_scores < self.threshold).all(axis=1)]
        elif self.method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            return X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            return X

# -------------------------------
# Example dataset
# -------------------------------
df = pd.DataFrame({
    'age': [25, 32, 47, 51, 62],
    'income': [50000, 60000, 55000, 75000, 90000],
    'gender': ['M','F','F','M','F'],        # nominal
    'region': ['North','South','East','West','North'], # nominal
    'education': ['high','medium','low','medium','high'] # ordinal
})

y = pd.Series([200, 250, 230, 300, 400])



# -------------------------------
# Select numeric and categorical columns
# -------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

ordinal_cols = ['education']
education_order = ['low', 'medium', 'high']

categorical_cols = [c for c in categorical_cols if c not in ordinal_cols]

# -------------------------------
# Numeric preprocessing
# -------------------------------
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('outlier', OutlierRemover(method='zscore', threshold=3)),
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
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(columns = 'y')   
y = train['y']          

X_train, y_train, X_val, y_val = train_test_split(X, test_size=0.2, random_state=42)


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
    mse_scores = cross_val_score(pipeline, X_sample, y_sample, cv=kf, scoring='neg_mean_squared_error') # or X_train, y_train
    rmse_scores = np.sqrt(-mse_scores)
    
    # CV R²
    r2_scores = cross_val_score(pipeline, X_sample, y_sample, cv=kf, scoring='r2', n_jobs=-1)
    
    # Append results for this model
    cv_results.append({
        'Model': name,
        'RMSE Mean': rmse_scores.mean(),
        'R2 Mean': r2_scores.mean(),
        'RMSE Std': rmse_scores.std(),
        'R2 Std': r2_scores.std()
    })

# Create DataFrame and sort
cv_df = pd.DataFrame(cv_results).sort_values(by='RMSE Mean')
print("Initial CV Results on Sample:\n", cv_df)

# Select best model based on RMSE and R2
best_model_name = cv_df.iloc[0]['Model']
best_model = models[best_model_name]

print('-' *100)
print(f"Best model selected: {best_model_name}")

# -------------------------------
# Fit best model on whole data set
# -------------------------------
best_model = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)])
        
pipeline.fit(X_train, y_train)

# -------------------------------
# Predictions on test set with baseline model
# -------------------------------
try: 
    y_pred = pipeline.predict(test)
    print("Predictions:", y_pred)
    pred_df = pd.DataFrame({'Predicted': test})
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
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_jobs=-1,
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
    print("Best CV R²:", random_search.best_score_)
    
    # Validation predictions
    y_pred_val = random_search.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    
    print(f"Validation RMSE: {np.sqrt(mse):,.2f}")
    print(f"Validation MAE: {mae:,.2f}")
    print(f"Validation R2 Score: {r2:.2f}")
    
    # Apply to test set
    y_pred_test = random_search.predict(test)
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
    2. Residuals vs Actual values
    3. Residual distribution (histogram + KDE)
    4. Line plot comparison (Actual vs Predicted sequence)
    """
    residuals = y_val - y_pred_val

    plt.figure(figsize=(14, 12))

    # 1️⃣ Actual vs Predicted
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=y_val, y=y_pred_val, color='royalblue', alpha=0.7)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2)
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual (y_val)")
    plt.ylabel("Predicted (y_pred_val)")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2️⃣ Residuals vs Actual
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=y_val, y=residuals, color='darkorange', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title("Residuals vs Actual Values")
    plt.xlabel("Actual (y_val)")
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
    plt.plot(y_pred_val, label='Predicted', color='dodgerblue', linestyle='--')
    plt.title("Actual vs Predicted (Sequence View)")
    plt.xlabel("Observation Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

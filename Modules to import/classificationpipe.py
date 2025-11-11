# ==============================================
# FULL CLASSIFICATION PIPELINE (Train + Validation)
# ==============================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Classification Models
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------------
# Custom Transformer: Outlier Remover (optional)
# -------------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
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
        elif self.method == 'iqr': # removes outliers beyond 1.5*IQR
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            return X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            return X

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
y = pd.Series([1, 0, 0, 1, 1])  # classification labels

# -------------------------------
# Column Selection
# -------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
ordinal_cols = ['education']
education_order = ['low', 'medium', 'high']
categorical_cols = [c for c in categorical_cols if c not in ordinal_cols]

# -------------------------------
# Preprocessing Pipelines
# -------------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('outlier', OutlierRemover(method='zscore', threshold=3)),
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
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1') # 'f1_weighted' for imbalanced datasets

    #acc_scores = cross_val_score(pipeline, X_sample, y_sample, cv=kf, scoring='accuracy')
    #f1_scores = cross_val_score(pipeline, X_sample, y_sample, cv=kf, scoring='f1')   

    # Optional: ROC-AUC (only for models with predict_proba)
    if hasattr(model, "predict_proba"):   # <-- should check model, not pipeline.named_steps
        y_proba_cv = cross_val_predict(pipeline, X_train, y_train, cv=skf, method='predict_proba')
        roc_auc = roc_auc_score(y_train, y_proba_cv[:, 1])
        print(f"{name} Cross-validated ROC-AUC: {roc_auc:.3f}")
    else:
        roc_auc = np.nan

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
    scoring='accuracy', # 'f1_weighted' for imbalanced datasets
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
print("Best CV Accuracy :", random_search.best_score_) 

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
# Visualization Function
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

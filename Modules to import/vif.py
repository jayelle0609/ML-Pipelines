import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

class MulticollinearityDetector:
    """
    Detect multicollinearity in NUMERICAL FEATURES using Variance Inflation Factor (VIF).
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        """
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
        self.vif_table = pd.DataFrame()
    
    def compute_vif(self, columns: list = None) -> pd.DataFrame:
        """
        Compute VIF for specified columns or all numeric columns by default,
        and print a comprehensive suggestion about multicollinearity.
        
        Parameters:
            columns (list, optional): List of column names to compute VIF. Defaults to all numeric.
            
        Returns:
            pd.DataFrame: Table with columns ['feature', 'VIF']
        """
        if columns is None:
            columns = self.numeric_cols
        X = self.df[columns].copy()
        
        # Compute VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        self.vif_table = vif_data
        
        # Print VIF table
        print("\nVIF Table:")
        print(vif_data)
        
        # Print consolidated suggestion block
        suggestion_text = """

VIF (Variance Inflation Factor) for a feature measures how much the variance of that feature’s coefficient is “inflated” 
due to linear correlation with all other features in the dataset.

- VIF < 5: No significant multicollinearity detected.
- VIF > 5: Moderate multicollinearity. 
- VIF > 10: High multicollinearity. 

Recommendations to address multicollinearity:
    • Ridge/Lasso regression to stabilize coefficients.
    • Feature engineering or combining correlated features.
    • Apply PCA to transform correlated features into uncorrelated components. 
    • High multicollinearity makes interpreting coefficients unstable.
    • The effect of a 1-unit increase of feature A on target variable cannot be quantified separately from the other variable it has high multicollinearity with.

"""
        print(suggestion_text)
        
        return vif_data

    def drop_high_vif(self, threshold: float = 5.0, columns: list = None) -> pd.DataFrame:
        """
        Iteratively drop columns with VIF above threshold.
        
        Parameters:
            threshold (float): VIF threshold above which to drop features.
            columns (list, optional): Columns to consider. Defaults to all numeric columns.
            
        Returns:
            pd.DataFrame: DataFrame with high-VIF features removed
        """
        if columns is None:
            columns = self.numeric_cols
        X = self.df[columns].copy()
        
        while True:
            vif_df = self.compute_vif(columns=X.columns.tolist())
            max_vif = vif_df['VIF'].max()
            if max_vif > threshold:
                drop_col = vif_df.sort_values('VIF', ascending=False)['feature'].iloc[0]
                print(f"Dropping '{drop_col}' with VIF={max_vif:.2f}\n")
                X = X.drop(columns=[drop_col])
            else:
                break
        return X


# -------------------------
# Example Usage
# -------------------------
df = pd.DataFrame({
    'age': [25, 32, 47, 51, 62],
    'income': [50000, 60000, 80000, 75000, 90000],
    'spending_score': [60, 65, 70, 72, 80],
    'savings': [10000, 15000, 20000, 18000, 22000]
})

# Initialize detector
mc = MulticollinearityDetector(df)

# Compute VIF for all numeric columns
print("VIF Table for all columns:")
mc.compute_vif()

# Optionally, drop features above a threshold
clean_df = mc.drop_high_vif(threshold=5.0)
print("\nDataFrame after removing high-VIF columns:")
print(clean_df)

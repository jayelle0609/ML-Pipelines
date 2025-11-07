import scipy.stats as stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class StatTests:
    """
    A collection of common statistical tests for numerical and categorical data.
    Each method prints results with interpretation.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        """
        self.df = df.copy()
        self.num_cols = df.select_dtypes(include='number').columns.tolist()
        self.cat_cols = df.select_dtypes(include='object').columns.tolist()

    # -------------------------------
    # 1. Pearson Correlation
    # -------------------------------
    def pearson_corr(self, col1: str, col2: str):
        r, p = stats.pearsonr(self.df[col1], self.df[col2])
        print(f"Pearson correlation r between '{col1}' and '{col2}': {r:.2f}, p-value: {p:.2f}")
        abs_r = abs(r)
        if abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        sig_text = "statistically significant" if p < 0.05 else "not statistically significant"
        print(f"Interpretation: {strength} correlation, {sig_text}.")
        return r, p

    # -------------------------------
    # 2. Spearman Rank Correlation
    # -------------------------------
    def spearman_corr(self, col1: str, col2: str):
        rho, p = stats.spearmanr(self.df[col1], self.df[col2])
        print(f"Spearman rho between '{col1}' and '{col2}': {rho:.2f}, p-value: {p:.4f}")
        abs_rho = abs(rho)
        if abs_rho < 0.3:
            strength = "weak"
        elif abs_rho < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        direction = "positive" if rho > 0 else "negative"
        significance = "statistically significant" if p < 0.05 else "not statistically significant"
        print(f"Interpretation: {strength} {direction} correlation, {significance}.")
        return rho, p

    # -------------------------------
    # 3. T-Test (Two-sample)
    # -------------------------------
    def t_test(self, col1: str, col2: str, equal_var=True):
        t_stat, p_val = stats.ttest_ind(self.df[col1], self.df[col2], equal_var=equal_var)
        print(f"T-test between '{col1}' and '{col2}': t={t_stat:.2f}, p-value={p_val:.2f}")
        interpretation = "reject null hypothesis: means are significantly different" if p_val < 0.05 else "fail to reject null hypothesis: means are not significantly different"
        print(f"Interpretation: {interpretation}.")
        return t_stat, p_val

    # -------------------------------
    # 4. ANOVA
    # -------------------------------
    def anova(self, target: str, group: str):
        groups = [self.df[target][self.df[group] == g] for g in self.df[group].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"ANOVA for '{target}' by '{group}': F={f_stat:.2f}, p-value={p_val:.2f}")
        interpretation = "At least one group mean is statistically significantly different. Difference is not due to random noise or chance." if p_val < 0.05 else "No statistically significant difference between group means. Difference due to random noise or chance."
        print(f"Interpretation: {interpretation}.")
        return f_stat, p_val

    # -------------------------------
    # 5. Chi-Square Test of Independence
    # -------------------------------
    def chi2_test(self, col1: str, col2: str):
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Chi2 test between '{col1}' and '{col2}': chi2={chi2:.2f}, p-value={p:.2f}, dof={dof}")
        interpretation = "variables are associated" if p < 0.05 else "variables are independent"
        print(f"Interpretation: {interpretation}.")
        return chi2, p, dof, expected

    # -------------------------------
    # 6. Normality Test (Shapiro-Wilk)
    # -------------------------------
    def shapiro_test(self, col: str):
        stat, p_val = stats.shapiro(self.df[col])
        print(f"Shapiro-Wilk test for '{col}': W={stat:.2f}, p-value={p_val:.2f}")
        interpretation = "data is normally distributed" if p_val > 0.05 else "data is not normally distributed"
        print(f"Interpretation: {interpretation}.")
        return stat, p_val

    # -------------------------------
    # 7. Levene’s Test for Equal Variances
    # -------------------------------
    def levene_test(self, col: str, group: str):
        groups = [self.df[col][self.df[group] == g] for g in self.df[group].unique()]
        stat, p_val = stats.levene(*groups)
        print(f"Levene test for '{col}' by '{group}': stat={stat:.2f}, p-value={p_val:.2f}")
        interpretation = "variances are equal across groups" if p_val > 0.05 else "variances are not equal across groups"
        print(f"Interpretation: {interpretation}.")
        return stat, p_val

    # -------------------------------
    # 8. Suggest which test to use
    # -------------------------------
    def which_test(self, col1: str, col2: str):
        if col1 in self.num_cols and col2 in self.num_cols:
            print(f"\nColumns '{col1}' and '{col2}' are numeric.")
            print("Suggested tests:")
            print(" - Parametric: Pearson correlation, t-test, ANOVA (if comparing groups)")
            print(" - Non-parametric: Spearman correlation, Mann-Whitney U, Kruskal-Wallis")
        elif col1 in self.cat_cols and col2 in self.cat_cols:
            print(f"\nColumns '{col1}' and '{col2}' are categorical.")
            print("Suggested test: Chi-Square Test of Independence")
        else:
            print(f"\nColumns '{col1}' and '{col2}' are of different types (numeric vs categorical).")
            print("Suggested test:")
            print(" - Numeric vs Categorical: t-test (2 groups), ANOVA (>2 groups), Mann-Whitney / Kruskal-Wallis for non-parametric")
        print("Always check distribution (normality) for numeric data before using parametric tests.")

    # -------------------------------
    # 9. Correlation matrix and heatmap
    # -------------------------------
    def visualize_corr_matrix(self, method='pearson', visualize=True):
        """
        Compute correlation matrix for all numeric columns with optional heatmap.
        """
        if method not in ['pearson', 'spearman']:
            raise ValueError("Method must be 'pearson' or 'spearman'")

        corr_matrix = self.df[self.num_cols].corr(method=method)
        print(f"\n{method.capitalize()} correlation matrix:")
        print(corr_matrix.round(2))

        if visualize:
            plt.figure(figsize=(8, 6))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask = mask, square=True)
            plt.title(f'{method.capitalize()} Correlation Heatmap')
            plt.show()

        return corr_matrix


# Example usage :
df = pd.DataFrame({
    'age': [25, 32, 47, 51, 62],
    'income': [50000, 60000, 80000, 75000, 90000],
    'spending_score': [60, 65, 70, 72, 80],
    'savings': [10000, 15000, 20000, 18000, 22000],
    'group': ['A', 'A', 'B', 'B', 'B']
})

# Initialize StatTests with the df of interest
stat_test = StatTests(df)

# Pearson correlation between two columns
stat_test.pearson_corr('age', 'income')

# Spearman correlation between two columns
stat_test.spearman_corr('age', 'income')

# Shapiro normality test
stat_test.shapiro_test('income')

# Suggest which test to use
stat_test.which_test('age', 'group')

# Correlation matrix for all numeric columns with heatmap
stat_test.visualize_corr_matrix(method='pearson')

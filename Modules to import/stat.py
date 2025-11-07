import scipy.stats as stats
import pandas as pd

class StatTests:
    """
    A collection of common statistical tests for numerical and categorical data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        """
        self.df = df.copy()
    
    # -------------------------------
    # 1. Pearson Correlation
    # -------------------------------
def pearson_corr(self, col1: str, col2: str):
    """
    Compute Pearson correlation coefficient between two numeric columns.
    Returns correlation coefficient and p-value, with a brief interpretation.
    """
    r, p = stats.pearsonr(self.df[col1], self.df[col2])
    print(f"Pearson correlation r between '{col1}' and '{col2}': {r:.2f}, p-value: {p:.2f}")
    
    # Interpretation based on correlation strength and significance
    if p < 0.05:
        sig_text = "statistically significant"
    else:
        sig_text = "not statistically significant"
        
    # Determine correlation strength
    abs_r = abs(r)
    if abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    print(f"Interpretation: {strength} correlation, {sig_text}.")
    
    return r, p

    # -------------------------------
    # 2. Spearman Rank Correlation
    # -------------------------------
def spearman_corr(self, col1: str, col2: str):
        """
        Compute Spearman rank correlation between two numeric columns.
        Returns correlation coefficient and p-value.
        """
        rho, p = stats.spearmanr(self.df[col1], self.df[col2])
        print(f"Spearman rho between '{col1}' and '{col2}': {rho:.4f}, p-value: {p:.4f}")
        return rho, p

    # -------------------------------
    # 3. T-Test (Two-sample)
    # -------------------------------
def t_test(self, col1: str, col2: str, equal_var=True):
        """
        Perform independent two-sample t-test between two numeric columns.
        Returns t-statistic and p-value.
        """
        t_stat, p_val = stats.ttest_ind(self.df[col1], self.df[col2], equal_var=equal_var)
        print(f"T-test between '{col1}' and '{col2}': t={t_stat:.4f}, p-value={p_val:.4f}")
        return t_stat, p_val

    # -------------------------------
    # 4. ANOVA
    # -------------------------------
def anova(self, target: str, group: str):
        """
        Perform one-way ANOVA: compares means of target variable across groups.
        target: numeric column
        group: categorical column
        """
        groups = [self.df[target][self.df[group] == g] for g in self.df[group].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"ANOVA for '{target}' by '{group}': F={f_stat:.4f}, p-value={p_val:.4f}")
        return f_stat, p_val

    # -------------------------------
    # 5. Chi-Square Test of Independence
    # -------------------------------
def chi2_test(self, col1: str, col2: str):
        """
        Perform Chi-Square test of independence between two categorical columns.
        Returns chi2 statistic, p-value, degrees of freedom, and expected frequencies.
        """
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Chi2 test between '{col1}' and '{col2}': chi2={chi2:.4f}, p-value={p:.4f}, dof={dof}")
        return chi2, p, dof, expected

    # -------------------------------
    # 6. Normality Test (Shapiro-Wilk)
    # -------------------------------
def shapiro_test(self, col: str):
        """
        Test if a numeric column is normally distributed using Shapiro-Wilk test.
        """
        stat, p_val = stats.shapiro(self.df[col])
        print(f"Shapiro-Wilk test for '{col}': W={stat:.4f}, p-value={p_val:.4f}")
        return stat, p_val

    # -------------------------------
    # 7. Levene’s Test for Equal Variances
    # -------------------------------
def levene_test(self, col: str, group: str):
        """
        Test for equal variances of a numeric column across groups.
        """
        groups = [self.df[col][self.df[group] == g] for g in self.df[group].unique()]
        stat, p_val = stats.levene(*groups)
        print(f"Levene test for '{col}' by '{group}': stat={stat:.4f}, p-value={p_val:.4f}")
        return stat, p_val

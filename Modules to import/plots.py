import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class SeabornPlots:
    """
    A module for common Seaborn plots with easy-to-use functions.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -------------------------------
    # Histogram
    # -------------------------------
    def histogram(self, col: str, bins: int = 10, kde: bool = True):
        plt.figure(figsize=(7, 5))
        sns.histplot(self.df[col], bins=bins, kde=kde, color='skyblue')
        plt.title(f'Histogram of {col}')
        plt.show()

    # -------------------------------
    # Boxplot
    # -------------------------------
    def boxplot(self, col: str, by: str = None):
        plt.figure(figsize=(7, 5))
        if by:
            sns.boxplot(x=by, y=col, data=self.df)
            plt.title(f'Boxplot of {col} by {by}')
        else:
            sns.boxplot(y=self.df[col])
            plt.title(f'Boxplot of {col}')
        plt.show()

    # -------------------------------
    # Violin Plot
    # -------------------------------
    def violinplot(self, col: str, by: str = None):
        plt.figure(figsize=(7, 5))
        if by:
            sns.violinplot(x=by, y=col, data=self.df)
            plt.title(f'Violin plot of {col} by {by}')
        else:
            sns.violinplot(y=self.df[col])
            plt.title(f'Violin plot of {col}')
        plt.show()

    # -------------------------------
    # Scatter Plot
    # -------------------------------
    def scatterplot(self, x: str, y: str, hue: str = None, size: str = None):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=x, y=y, hue=hue, size=size, data=self.df)
        plt.title(f'Scatter plot of {y} vs {x}')
        plt.show()

    # -------------------------------
    # Pairplot
    # -------------------------------
    def pairplot(self, hue: str = None, diag_kind: str = 'kde'):
        sns.pairplot(self.df, hue=hue, diag_kind=diag_kind)
        plt.suptitle('Pairplot of all numeric features', y=1.02)
        plt.show()

    # -------------------------------
    # Heatmap
    # -------------------------------
    def heatmap(self, cols: list = None, annot: bool = True, cmap: str = 'coolwarm'):
        if cols is None:
            cols = self.df.select_dtypes(include='number').columns.tolist()
        corr_matrix = self.df[cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap=cmap, square=True)
        plt.title('Correlation Heatmap')
        plt.show()
        return corr_matrix

    # -------------------------------
    # Countplot
    # -------------------------------
    def countplot(self, col: str, hue: str = None):
        plt.figure(figsize=(7, 5))
        sns.countplot(x=col, hue=hue, data=self.df)
        plt.title(f'Countplot of {col}')
        plt.show()

    # -------------------------------
    # Barplot
    # -------------------------------
    def barplot(self, x: str, y: str, hue: str = None, ci: str = "sd"):
        plt.figure(figsize=(7, 5))
        sns.barplot(x=x, y=y, hue=hue, ci=ci, data=self.df)
        plt.title(f'Barplot of {y} by {x}')
        plt.show()

    # -------------------------------
    # Lineplot
    # -------------------------------
    def lineplot(self, x: str, y: str, hue: str = None):
        plt.figure(figsize=(7, 5))
        sns.lineplot(x=x, y=y, hue=hue, data=self.df)
        plt.title(f'Lineplot of {y} vs {x}')
        plt.show()

    # -------------------------------
    # Regression plot (regplot)
    # -------------------------------
    def regplot(self, x: str, y: str):
        plt.figure(figsize=(7, 5))
        sns.regplot(x=x, y=y, data=self.df, scatter_kws={'s':50}, line_kws={'color':'red'})
        plt.title(f'Regression plot of {y} vs {x}')
        plt.show()

    # -------------------------------
    # LM Plot (regression with hue/group)
    # -------------------------------
    def lmplot(self, x: str, y: str, hue: str = None, col: str = None, row: str = None, scatter_kws=None, line_kws=None):
        sns.lmplot(x=x, y=y, hue=hue, col=col, row=row, data=self.df,
                   scatter_kws=scatter_kws if scatter_kws else {'s':50},
                   line_kws=line_kws if line_kws else {'color':'red'})
        plt.suptitle(f'LM plot of {y} vs {x}', y=1.02)
        plt.show()

    # -------------------------------
    # KDE Plot
    # -------------------------------
    def kdeplot(self, col: str, hue: str = None, fill: bool = True):
        plt.figure(figsize=(7, 5))
        sns.kdeplot(data=self.df, x=col, hue=hue, fill=fill)
        plt.title(f'KDE plot of {col}')
        plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

################################################################################
############################### Set Theme and Style ############################
################################################################################
sns.set_style("whitegrid")  # options: darkgrid, white, ticks, etc.
sns.set_context("notebook") # options: talk, paper, poster, notebook

################################################################################
############################### 1. Relational Plots ############################
################################################################################
# Scatter plot
sns.scatterplot(x='x_col', y='y_col', hue='category', style='category', size='size_col', palette='deep', data=df)

# Line plot
sns.lineplot(x='x_col', y='y_col', hue='category', style='category', ci=95, data=df)

# Figure-level relational plot (scatter/line) with facets
sns.relplot(x='x_col', y='y_col', hue='category', kind='scatter', col='col_facet', row='row_facet', data=df) # kind = 'line'

################################################################################
############################### 2. Distribution Plots ##########################
################################################################################
# Histogram
sns.histplot(x='col', bins=10, hue='category', multiple='stack', kde=True, data=df)

# Looping thru numeric cols to check for distribution type
# NUMERIC PLOTS NUMERIC NUMERIC NUMERIC NUMERIC
for cols in df.select_dtypes(include = np.number).columns:
  plt.figure(figsize = (3,3))
  sns.histplot(data = df, x = cols, kde = True, hue = 'col category of itnerests') # OR
  sns.displot(x='col', hue='category', kind='hist', col='col_facet', row='row_facet', data=df) # KINDA BETTER
  plt.title(f"Distribution of {cols}")

# Kernel Density Estimate
sns.kdeplot(x='col', y=None, hue='category', fill=True, bw_adjust=1, data=df)

# Rug plot (individual observations)]
# not very helpful
sns.rugplot(x='col', height=0.05, data=df)

# Figure-level histogram/kde
# YO VERY GOOD!!!
sns.displot(x='col', hue='category', kind='hist', col='col_facet', row='row_facet', data=df)

################################################################################
############################### 3. Categorical Plots ###########################
################################################################################
sub_cat = df['col2']
for cols in df.select_dtypes(exclude = np.number):
  plt.figure(figsize = (3,3))
  sns.countplot(x= cols, hue = sub_cat, data = df)

cols_of_interest = ['sex', 'smoker',	'day',	'time',	 'size']
for cols in cols_of_interest:
  sns.countplot(tips, x=cols, hue = 'time')

cols = df.select_dtypes(include=np.number).columns
fig, axes = plt.subplots(2, 2, figsize=(10,8))  # 2 rows, 2 columns
axes = axes.flatten()  # Flatten to easily iterate
for ax, col in zip(axes, cols[:4]):  # Only first 4 numerical columns
    sns.histplot(data=tips, x=col, kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Box plot
sns.boxplot(x='category', y='value', hue='sub_category', orient='v', notch=False, data=df)

# Violin plot
sns.violinplot(x='category', y='value', hue='sub_category', split=True, inner='quartile', data=df)

# Strip plot
sns.stripplot(x='category', y='value', hue='sub_category', jitter=True, dodge=True, data=df)

# Swarm plot
sns.swarmplot(x='category', y='value', hue='sub_category', data=df)

# Bar plot (mean + confidence interval)
sns.barplot(x='category', y='value', hue='sub_category', ci=95, estimator=sum, data=df)

# Count plot (frequency of each category)
sns.countplot(x='category', hue='sub_category', data=df)

# Figure-level categorical plot
sns.catplot(x='category', y='value', hue='sub_category', kind='box', col='col_facet', row='row_facet', data=df)

################################################################################
############################### 4. Matrix / Correlation Plots ##################
################################################################################
# Heatmap (e.g., correlation matrix)
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(tips.select_dtypes(include = np.number).corr()))
sns.heatmap(data = tips.select_dtypes(include = np.number).corr(), annot = True, fmt = '.2f', cmap = 'coolwarm', mask = mask)
plt.title("Correlation Heatmap (lower triangle only)")
plt.show()

# Clustermap (hierarchical clustering)
sns.clustermap(df.corr(), method='ward', metric='euclidean', standard_scale=1, cmap='viridis')

################################################################################
############################### 5. Regression / Model Plots ###################
################################################################################
# Axes-level regression plot
sns.regplot(x='x_col', y='y_col', data=df, ci=95, scatter_kws={'s':50}, line_kws={'color':'red'})
#  YO THIS IS SO GOOD, IS A SCATTER PLOT WITH A BEST FIT LINE!!!

# Figure-level regression plot with facets
sns.lmplot(x='x_col', y='y_col', hue='category', col='col_facet', row='row_facet', data=df, ci=95)

# Residual plot
sns.residplot(x='x_col', y='y_col', lowess=True, data=df)

################################################################################
############################### 6. Pair / Joint Plots ##########################
################################################################################
# Pairplot (scatter + histogram/kde for all numeric cols)
sns.pairplot(df, hue='category', diag_kind='hist', kind='scatter', corner=False)

# Jointplot (scatter + marginals)
sns.jointplot(x='x_col', y='y_col', kind='scatter', hue='category', data=df)

################################################################################
############################### 7. Timeseries / Facet Grids ###################
################################################################################
# Relational plot with line and facets
sns.relplot(x='time', y='value', hue='category', kind='line', col='col_facet', row='row_facet', data=df)

# Categorical plot with facets
sns.catplot(x='category', y='value', hue='sub_category', kind='bar', col='col_facet', row='row_facet', data=df)

################################################################################
################################### Show Plot ##################################
################################################################################
plt.show()


plt.tight_layout() # Adjust subplots to fit into figure area.
plt.xlabel('X-axis Label') # Set x-axis label.
plt.ylabel('Y-axis Label') # Set y-axis label.
plt.title('Plot Title') # Set plot title.
plt.legend(title='Legend Title') # Add legend with title.
plt.grid(True) # Enable grid.

plt.annotate('Annotation Text', xy=(x, y), xytext=(x_offset, y_offset),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, color='red')

################################################################################
################################### Highlight ##################################
################################################################################

# Highlight x between 3 and 6
# Highlight a verical range 
plt.plot(x, y)
plt.axvspan(3, 6, color='yellow', alpha=0.3)

# Highlight a horizontal range
# y between 0.5 and 1
plt.plot(x, y)
plt.axhspan(0.5, 1, color='green', alpha=0.2)


# Highlight a region under curve 
# Highlight region where x is between 2 and 5
plt.plot(x, y)
plt.fill_between(x, y, where=(x>=2) & (x<=5), color='red', alpha=0.3)


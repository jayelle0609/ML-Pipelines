import pandas as pd 


# -------------------------------
# Used to combine two DataFrames based on keys/columns, similar to SQL joins
# -------------------------------
pd.merge(left = left_df, right = right_df,
         how='inner',   # type of join: 'inner', 'outer', 'left', 'right'
         on=None,       # column(s) to join on
        )

# -------------------------------
# Used to stack/append multiple DataFrames vertically or horizontally
# -------------------------------
pd.concat(objs=[df1, df2, ...],   # list of DataFrames
          axis=0,                 # 0 = stack rows, 1 = stack columns 
          join='outer',           # 'outer'=union of columns, 'inner'=intersection
          ignore_index=True,     # repeated index if False
         )

df.append(other=df2, ignore_index=True) # append df2 rows to df


df.groupby('dept')['salary'].agg(['mean','max'])

df.groupby('department')['salary'].mean()

df.sort_values(by='age', ascending=False)

df.sort_index()

df['income'].fillna(0)

df.pivot(index='id', # old
         columns='month', # old
         values='sales') # reshape from long to wide #old

# reshape from wide to long
df.melt(id_vars='id', # old 
        value_vars=['Jan','Feb'], # unique col values, from old bring ot new
        var_name = 'month', # unique col name, new
        value_name = 'sales') # name of next new col to show values,

df.pivot_table(index='dept', # groyp by unique values in what col
               values='salary', # the values u want
               aggfunc='mean') # the agg type

df['name'].str.lower().str.strip()

df[df['name'].str.contains('Bob')] # gives u row values of that filter 
df['name'] = df['name'].str.replace('Bob', 'Robert', regex=True) 
# replaces Bob with Robert with regex
# must be BOB exactly

df.loc[df['name'].str.contains('Bob', na=False, case = False), 'name'] = 'Bob'
# If a value is NaN, treat it as False for the condition. As na values will throw an error to the formula
# case insensitive search
# replaces entire cells with any Bob in the string with Bob only

df = df.rename(columns={'name': 'Name'}
# renames cols

df['name'].str.replace('Alice','Ally') # must be alice exactly

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year # extract year from datetime
df['month'] = df['date'].dt.month # extract month from datetime


df['age'].apply(lambda x: x+1) # apply function to each element in column

df.replace({'yes':1,'no':0})

pd.get_dummies(df, drop_first=True)

df.map({'Male':1,'Female':0})

df['age'].plot(kind='hist', color = "orange") # quick plot
# kind = 'hist', 'line', 'bar', 'barh' , 'box', 'pie', 'area'

df['sales'].plot(kind='pie')
df['sales'].plot(kind='area', alpha = 0.4)

df = pd.DataFrame({'Age': [23, 45, 31, 37, 29, 41]})
df['Age'].plot(kind='hist', bins=5, color='green', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()


df = pd.DataFrame({
    'Math': [80, 90, 70, 85, 95],
    'Science': [75, 85, 80, 90, 100]
})

df.plot(kind='box', color='blue')
plt.title('Score Distribution')
plt.show()


df.set_index('Month').plot(kind='area', alpha=0.5)
plt.title('Monthly Product Sales')
plt.ylabel('Sales')
plt.show()

df = pd.DataFrame({
    'Height': [150, 160, 170, 180, 175],
    'Weight': [50, 60, 65, 80, 75]
})

df.plot(kind='scatter', x='Height', y='Weight', color='red')
plt.title('Height vs Weight')
plt.show()


df = pd.DataFrame({'Category': ['A','B','C'], 'Values':[30, 45, 25]})
df.set_index('month')['sales'].plot(kind='pie', autopct='%1.1f%%', startangle=90)
# group by month and show sales value on the pie
plt.ylabel('')
plt.title('Category Share')
plt.show()


data = df.corr()
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title("Heatmap of XXX")
plt.show()


# Take a random 10% of the DataFrame
df_sample = df.sample(frac=0.1, random_state=42)  # random_state for reproducibility

# Stratify to ensure original class proportions are preserved 
from sklearn.model_selection import train_test_split
# Suppose y is your imbalanced target
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,      # <-- ensures class proportions are preserved
    random_state=42
)

# if you're doing cv
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
X = df.drop('target', axis=1)
y = df['target']
# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier()
scoring = ['accuracy', 'roc_auc', 'f1']
scores = cross_validate(model, X, y, cv=skf, scoring=scoring)
print("CV scores:", scores)
# Print the multiple scoring metrics
for metric in scoring:
    print(f"{metric}: {scores['test_' + metric]}")
    print(f"Mean {metric}: {scores['test_' + metric].mean():.2f}\n")

df['gender'].value_counts()

df[df['age'] > 30] # filter rows based on condition

df[df['income'].isnull()] # filter rows with missing values in 'income' column
# so u will see rows with NaN in income column

df.drop(columns=['unnecessary_col1', 'unnecessary_col2'], inplace=True)

df.iloc[0:5, 1:3] # select by position: rows 0-4, columns 1-2

df.loc[df['age'] > 30, ['name', 'income']] # select by label with condition, give me cols name and income for those that are above 30

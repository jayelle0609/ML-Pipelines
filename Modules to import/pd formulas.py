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

df.pivot(index='id', 
         columns='month', 
         values='sales') # reshape from long to wide

df.melt(id_vars='id', 
        value_vars=['Jan','Feb']) # reshape from wide to long

df.pivot_table(index='dept', 
               values='salary', 
               aggfunc='mean')

df['name'].str.lower().str.strip()

df[df['name'].str.contains('Bob')]

df['name'].str.replace('Alice','Ally')

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year # extract year from datetime
df['month'] = df['date'].dt.month # extract month from datetime


df['age'].apply(lambda x: x+1) # apply function to each element in column

df.replace({'yes':1,'no':0})

df.map({'Male':1,'Female':0})

df['age'].plot(kind='hist', color = "orange") # quick plot
# kind = 'hist', 'line', 'bar', 'barh' , 'box'

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
df.set_index('Category')['Values'].plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.ylabel('')
plt.title('Category Share')
plt.show()


df.corr()


# Take a random 10% of the DataFrame
df_sample = df.sample(frac=0.1, random_state=42)  # random_state for reproducibility

df['gender'].value_counts()

df[df['age'] > 30] # filter rows based on condition

df[df['income'].isnull()] # filter rows with missing values in 'income' column
# so u will see rows with NaN in income column

df.drop(columns=['unnecessary_col1', 'unnecessary_col2'], inplace=True)

df.iloc[0:5, 1:3] # select by position: rows 0-4, columns 1-2

df.loc[df['age'] > 30, ['name', 'income']] # select by label with condition

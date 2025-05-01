# Converted from ass2.ipynb

# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# Load Dataset
df = pd.read_csv('StudentsPerformance (1).csv')

#  Display the first few rows of the data
print("First 5 rows of the dataset:\n")
print(df.head())


# Scan Variables for Missing Values and Inconsistencies
# --------------------------------------------
print("\nChecking for missing values:\n")
print(df.isnull().sum())


numeric_columns = ['Math_Score', 'Reading_Score', 'Writing_Score']

print(df.columns)


# Create the boxplot for the correct column name
sns.boxplot(x=df['Math_Score'])  # Use the correct column name here
plt.show()

# Detect outliers using Z-score method
z_scores = np.abs(stats.zscore(df[numeric_columns]))
outliers = (z_scores > 3)

print("\nNumber of outliers detected in each column:\n")
print(pd.Series(np.sum(outliers, axis=0), index=numeric_columns))
print("\n Outliers in Reading_Score:\n ")
sns.boxplot(x=df['Reading_Score'])
plt.show()

# Handle outliers: Remove rows where any z-score > 3
df_no_outliers = df[(z_scores < 3).all(axis=1)]

print("\nShape after removing outliers:", df_no_outliers.shape)

# Apply Data Transformation
# Transform 'math score' to reduce skewness using Log Transformation

# Note: Since log(0) is undefined, we add 1 before log transformation
df_no_outliers['Math_Score (log transformed)'] = np.log(df_no_outliers['Math_Score'] + 1)

# Plot original vs transformed distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_no_outliers['Math_Score'], kde=True, color='blue')
plt.title('Original Math Score Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df_no_outliers['Math_Score (log transformed)'], kde=True, color='green')
plt.title('Log-Transformed Math Score Distribution')

plt.show()



 # Final Dataset Information
print("\nFinal Dataset after cleaning and transformation:\n")
print(df_no_outliers.head())

# Also check new data types
print("\nData types after transformation:\n")
print(df_no_outliers.dtypes)


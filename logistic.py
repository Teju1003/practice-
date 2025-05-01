# Converted from logistic.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import pandas as pd

# Use raw string or escape backslashes properly
df = pd.read_csv(r'C:\Users\tejug\Downloads\Social_Network_Ads.csv')

# Check the data
print(df.head())
print(df.columns)


# Print shape and first few rows
print("Shape of dataset:", df.shape)
print(df.head())

print(df.head(5))

print(df.isnull().sum())

print(df.describe())

df.shape

# Select features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']



# Select features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']



# Proceed with train-test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Output results
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Error Rate: {1 - accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")



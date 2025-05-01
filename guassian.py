# Converted from guassian.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv('IRIS.csv')

print(df.head(5))

print(df.isnull().sum())

x = df.drop('species', axis=1)
y = df['species']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=0)

model = GaussianNB()

model.fit(x_train, y_train)

y_pred =model.predict(x_test)

y_pred

cm = confusion_matrix(y_test, y_pred)

print("/nconfision matrix:" ,cm)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')


print(f"\nAccuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

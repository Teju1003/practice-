# Converted from survived.ipynb

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv')

df



titanic = sns.load_dataset('titanic')

plt.figure(figsize=(10,6))
sns.boxplot( x='sex', y='age', hue='survived', data=titanic)
plt.title('Age Distribution by Gender and Survival Status on Titanic')
plt.xlabel('sex')
plt.ylabel('age')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot( x='sex', y='age', hue='survived', data=titanic)
plt.title('Age Distribution by Gender and Survival Status on Titanic')
plt.xlabel('sex')
plt.ylabel('age')
plt.show()


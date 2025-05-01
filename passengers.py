# Converted from passengers.ipynb

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('titanic.csv')

df

print(df.isnull().sum())

titanic = sns.load_dataset('titanic')

plt.figure(figsize=(10,6))
sns.histplot(data=titanic, x='fare', kde=True, bins=30, color='blue')
plt.title('price of fare')
plt.xlabel('fare')
plt.ylabel('no.of passenger')
plt.show()


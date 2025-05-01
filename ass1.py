# Converted from ass1.ipynb

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


df=pd.read_csv("iris.csv")

df.head()

df.describe()

df.shape

df.isnull().sum()

df.notnull()

df.dtypes

df['Species']=df['Species'].astype('category')
df.dtypes

df['Species_Encoded']=df['Species'].cat.codes
df.head()

Scaler=MinMaxScaler()

numeric_columns=['SepalLenghthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
df.head()

df.mean(numeric_only=True)

df.mode()

df.max(numeric_only=True)

col=['SepalLengthCm','PetalWidthCm','SepalWidthCm']
df.boxplot(col)
plt.show()


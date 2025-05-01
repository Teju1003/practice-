# Converted from ass4 .ipynb

import pandas as pd
import numpy as np
df=pd.read_csv("BostonHousing (1).csv")

df

df.isnull().sum()


corr = df.corr()
corr.shape

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,10))
sns.heatmap(df.corr().round(2), annot=True,cmap='coolwarm', linewidth=0.2 ,square=2)


plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

df.columns

x=df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
y=df['medv']

reg=LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=4)

reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

from sklearn import metrics

mean_squared_error(y_test,y_pred)

mean_absolute_error(y_test,y_pred)

# Scatter plot of Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Home Prices')
plt.show()



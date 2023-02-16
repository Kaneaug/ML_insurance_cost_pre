import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Data read and explored
df = pd.read_csv(r'C:\Users\kanem\Documents\ML_insurance_cost_pre\data\insurance.csv')
df.shape
df.info
df.describe()

# finding missing values

df.isnull().sum()

# Distribution of age plot

sns.set()
plt.figure(figsize=(6,6))
sns.displot(df['age'])
plt.title('Distribution of Age')
plt.show()

#Sex distribution plot 
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=df)
plt.title('Distribution of Sex')
df['sex'].value_counts()

#BMI distribution plot

plt.figure(figsize=(6,6))
sns.distplot(df['bmi'])
plt.title('BMI dist plot')
plt.show()

# Distribution of children

plt.figure((6,6))
sns.countplot(x='children', data=df)
plt.title('Children')
plt.show()

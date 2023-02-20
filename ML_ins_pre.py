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
df

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

plt.figure(figsize=(6,6))
sns.countplot(x='children', data=df)
plt.title('Children')
plt.show()

#Dist of smokers 

plt.figure(figsize=(8,8))
sns.countplot(x='smoker', data=df)
plt.title('Dist of smokers')
plt.show()

df['smoker'].value_counts()

# Dist of Region 

plt.figure(figsize=(8,8))
sns.countplot(x='region',data=df)
plt.title('Dist of regions')
plt.show()

df['region'].value_counts()

#Dist of charges

plt.figure(figsize=(8,8))
sns.distplot(df['charges'])
plt.title('Distribution of charges')
plt.show()

# Encoding catagrical features

df.replace({'sex':{'male':0,'female':1}}, inplace=True)
df.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

# Split Features and Target

X = df.drop(columns='charges', axis=1)
Y = df['charges']

X
Y

# Test Train Data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.2,random_state=2)

X.shape, X_train.shape, X_test.shape

# Regression model and r2 score on training data

linr = LinearRegression()
linr.fit(X_train,Y_train)
linr.coef_
train_predict = linr.predict(X_train)
r2_train = metrics.r2_score(Y_train,train_predict)
r2_train

# Regression model and r2 score on testing data

test_predict = linr.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_predict)
r2_test

# Build the predictive model

input_data = (31,1,35.74,0,0,0)
input_to_np_array = np.asarray(input_data)
input_reshape = input_to_np_array.reshape(1,-1)
input_reshape
prediction = linr.predict(input_reshape)
prediction

print('The insurance cost is USD ', prediction[0])
X
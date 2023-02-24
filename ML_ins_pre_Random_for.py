import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
sns.set_theme(style="whitegrid")

# Data read and explored
df = pd.read_csv(r'C:\Users\kanem\Documents\ML_insurance_cost_pre\data\insurance.csv')

# Encoding categorical features
df.replace({'sex': {'male':0, 'female':1}}, inplace=True)
df.replace({'smoker': {'yes':0, 'no':1}}, inplace=True)
df.replace({'region': {'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)

# Split Features and Target
X = df.drop(columns='charges', axis=1)
Y = df['charges']

X
Y
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.2,random_state=2)

# Build the Random Forest model
rfr = RandomForestRegressor(n_estimators=100, random_state=2)
rfr.fit(X_train, Y_train)

# Predict on test data and calculate R-squared score
test_predict = rfr.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_predict)
print("R-squared score on testing data: ", r2_test)

# Predict on new data point
input_data = (31,1,35.74,0,0,0)
input_to_np_array = np.asarray(input_data)
input_reshape = input_to_np_array.reshape(1,-1)
prediction = rfr.predict(input_reshape)
print('The insurance cost is USD ', prediction[0])
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('merged.csv')
# X = dataset.iloc[:, 3:].values
# y = dataset.iloc[:, 7].values
df3 = dataset[dataset.ResponseFlag == 1]
y = df3.iloc[:, 7].values
df4 = df3.drop ('Segment', axis = 1)
df5 = df4.drop('TotalResponseSales', axis = 1)
df6 = df5.drop('RECENCY', axis = 1)
df7 = df6.drop('BIRTH_DATE', axis = 1)
df8 = df7.drop('REC_MCHDY', axis = 1)
df9 = df8.drop('GENDER', axis = 1)
X = df9.iloc[:, :].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 198] = labelencoder_X.fit_transform(X[:, 198])
onehotencoder = OneHotEncoder(categorical_features = [198])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# Getting rid of multicollinearity among dummy variables
X = X[:, 1:]
X = np.nan_to_num(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

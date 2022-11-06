# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Churn_Modelling.csv")

#Independent Variable Matrix/ Vector
X = dataset.iloc[:,3:13].values

#Making Dependent Variable Matrix/ Vector
y= dataset.iloc[:, 13].values

#Encoding/Labeling Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#encoding the coutnry
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
#encoding the genders
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

X = X[:, 1:]

#Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Creating the ANN
#Importing the ANN Libraries and packages

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

#Addin the output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Completing the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs = 100)

#Making prediction and Evaluating the model

#Predicting Single Value/ new result with regression
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





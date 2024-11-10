#downloading the necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

#downloading the iris data set
df=load_iris()
print(df.feature_names)
print(df.target_names)
print(df.target)

#Separation of the data set into training and test data sets
X=df.data
y=df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Attribute scaling process
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


mlp=MLPClassifier(solver='lbfgs', alpha=1e-5, activation='tanh', 
                  hidden_layer_sizes=(10,), random_state=1)
mlp.fit(X_train, y_train)
predict=mlp.predict(X_test)

#Calculation of Errors
print("MAE=%0.4f"%mean_absolute_error(y_test, predict))
print("MSE=%0.4f"%mean_squared_error(y_test, predict))
print("MedAE=%0.4f"%median_absolute_error(y_test, predict))
print("Belirleme Katsayısı(R^2)=%0.4f"%r2_score(y_test, predict))
print("RMSE=%0.4f"%np.sqrt(mean_squared_error(y_test, predict)))
print("MAPE=%0.4f"%(mean_absolute_percentage_error(y_test, predict)*100),"%")
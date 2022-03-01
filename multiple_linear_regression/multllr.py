import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
x=ct.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=1)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train,y_train)

y_pred = regression.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1))





#%%


import matplotlib.pyplot as plt
import pandas
import numpy as np
import pickle
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('taxi.csv.txt')
#print(data.head())

data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values

X_train,X_test,Y_train,Y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)

reg = LinearRegression()
reg.fit(X_train,Y_train)
print("Train score: ",reg.score(X_train,Y_train))
print("Test score: ",reg.score(X_test,Y_test))
#model_creation

pickle.dump(reg, open('taxi.pkl','wb'))

model = pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80, 1770000, 6000, 85]]))

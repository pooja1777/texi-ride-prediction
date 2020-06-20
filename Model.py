import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('taxi.csv')
# print(data.head())
data_x=data.iloc[:,0:-1].values
data_y=data.iloc[:,-1].values

# print(data_x)
# print(data_y)

x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)

reg = LinearRegression()
reg.fit(x_train,y_train)

# y_pred=reg.predict(x_test)
# print(y_pred)
# print('\n\n')
# print(y_test)
print("Training : ",reg.score( x_train,y_train)) #To check score of training
print("Testing : ",reg.score(x_test,y_test))


pickle.dump(reg,open('model.pkl','wb')) # This will create a file model and save the model in it.

model=pickle.load(open('model.pkl','rb')) #Loading the model
print("Prediction is: ", model.predict([[80,1770000,6000,85]])) #Using the model

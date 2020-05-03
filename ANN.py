import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
import sklearn.metrics
import pickle

df=pd.read_csv("Data\Real_Combine.csv")
df=df.dropna()
print(df.head())
#sns.pairplot(df)
#plt.show()
print(df.corr())

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

print(X.shape)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)



NN_model=Sequential()
NN_model.add(Dense(128,kernel_initializer="normal",input_dim=x_train.shape[1],activation="relu"))
NN_model.add(Dense(256,kernel_initializer="normal",activation="relu"))
NN_model.add(Dense(256,kernel_initializer="normal",activation="relu"))
NN_model.add(Dense(256,kernel_initializer="normal",activation="relu"))

NN_model.add(Dense(1,kernel_initializer="normal",activation='linear'))
NN_model.compile(loss="mean_absolute_error",optimizer="adam",metrics=["mean_absolute_error"])
NN_model.summary()

model_history=NN_model.fit(x_train,y_train,validation_split=0.33,batch_size=10,nb_epoch=100)

#model evaluation
prediction=NN_model.predict(x_test)

sns.distplot(y_test.values.reshape(-1,1)-prediction)
plt.show()
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

NN_model.save("ANN_Regression.h5")

# Loading model to compare the results

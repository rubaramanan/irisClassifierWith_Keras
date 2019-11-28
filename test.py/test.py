import pandas as pd
import numpy as np
from tensorflow import keras


iris = pd.read_csv('IrisData.csv')

iris_features = iris.values[:, 1:5].astype('f')
iris_label = iris.values[:, 5]

for i in range(iris_label.size):
    if iris_label[i]=='Iris-setosa':
        iris_label[i]=0
    elif iris_label[i]=='Iris-versicolor':
        iris_label[i]=1
    else:
        iris_label[i]=2


x = keras.utils.to_categorical(iris_label)



model = keras.models.Sequential()

model.add(keras.layers.Dense(32,activation='relu',input_shape=(4,)))
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(iris_features,x,epochs=2000)


model.save('mymodel.h5')
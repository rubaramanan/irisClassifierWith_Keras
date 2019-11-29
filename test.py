import pandas as pd
import numpy as np
from tensorflow import keras


model = keras.models.load_model('mymodel.h5')

test = ([[5.9,3.0,5.1,1.8]])

result = np.argmax(model.predict(test))

iris_labes=['Iris-setosa','Iris-versicolor','Iris-virginica']

print(iris_labes[result])

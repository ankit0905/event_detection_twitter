from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from load_dataset import load_dataset
import numpy as np


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

input_dimension = 300
split_ratio = 0.20
batch_size = 100

# load dataset
X, y, num_classes = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# model creation
model = Sequential()
model.add(Dense(120, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.05, epochs=16, batch_size=batch_size, verbose=2)

print(model.summary())

prediction = model.predict(X_test)
# print(list(prediction))

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print("Accuracy: ", scores[1])
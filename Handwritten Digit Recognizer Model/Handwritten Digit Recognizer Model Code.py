import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers

with open('Handwritten Digits\emnist-digits-train-images-idx3-ubyte', 'rb') as f:
    x_train = np.fromfile(f, np.int8)
x_train = x_train[16:]
x_train = x_train.reshape([-1, 28, 28, 1])
#print(x_train.shape)
with open('Handwritten Digits\emnist-digits-train-labels-idx1-ubyte', 'rb') as f:
    y_train = np.fromfile(f, np.int8)
y_train = y_train[8:]

with open('Handwritten Digits\emnist-digits-test-images-idx3-ubyte', 'rb') as f:
    x_test = np.fromfile(f, np.int8)
x_test = x_test[16:]
x_test = x_test.reshape([-1, 28, 28, 1])

with open('Handwritten Digits\emnist-digits-test-labels-idx1-ubyte', 'rb') as f:
    y_test = np.fromfile(f, np.int8)
y_test = y_test[8:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=([28, 28, 1]))) #input shape removed if break then finger out the reshaping
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#made sparse
model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

hist = model.fit(x_train, y_train,batch_size=128,epochs=3,verbose=1,validation_data=(x_test, y_test))
print("Done Training Model")
model.save('mnist2.h5')
print("Saving the model as mnist.h5")


score = model.evaluate(x_test, y_test, verbose=0)
print('Final Test loss:', score[0])
print('Final Test accuracy:', score[1])
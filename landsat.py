import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

samplePath = '/Users/cfarzaneh/Desktop/8class_new60percent/8classes_with_path_dim_9/'

x = np.load(samplePath+'X_train_patch.npy','r')
y = np.load(samplePath+'Y_train_patch.npy','r')

model = Sequential()
model.add(Conv2D(8,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(9,9,8)))
model.add(Conv2D(16,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(32,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(128,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(3200, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=Adam(lr=0.0001),
	metrics=['accuracy'])

model.fit(x,y,batch_size=10,epochs=2,verbose=1,validation_split=0.2)
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

dataPath = '/home/cameron/Desktop/8class_new60percent/8classes_with_path_dim_'

for i in range(0,2):
	if i == 0:
		samplePath = dataPath + '5/'
	elif i == 1:
		samplePath = dataPath + '9/'

	x = np.load(samplePath+'X_train_patch.npy')
	y = np.load(samplePath+'Y_train_patch.npy')

	valLabels = []
	valSamples = []
	for file in sorted(os.listdir(samplePath)):
		if file.startswith('test_label'):
			valLabels.append(np.load(samplePath+file))
		elif file.startswith('test_patch'):
			valSamples.append(np.load(samplePath+file))

	valLabels = np.concatenate(valLabels,axis=0)
	valSamples = np.concatenate(valSamples,axis=0)

	if i == 0:
		inputShape = (5,5,8)
	elif i == 1:
		inputShape = (9,9,8)

	model = Sequential()
	model.add(Conv2D(8,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=inputShape))
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

	epochs = 30
	hs = model.fit(x,y,batch_size=10,epochs=epochs,verbose=1,validation_data=(valSamples,valLabels))

	score = model.evaluate(valSamples,valLabels,verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, epochs), hs.history["loss"], label="train_loss")
	plt.plot(np.arange(0, epochs), hs.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, epochs), hs.history["acc"], label="train_acc")
	plt.plot(np.arange(0, epochs), hs.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper left")

	if i == 0:
		plt.savefig("results_5dim.png")
	elif i == 1:
		plt.savefig("results_9dim.png")

	model = None

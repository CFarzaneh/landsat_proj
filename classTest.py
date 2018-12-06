import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
import os


for dim in ['3','5','7','9','11']:

	samplePath = '8class_new60percent/8classes_with_path_dim_' + dim + '/'

	model = load_model('model_'+dim+'dim.h5')
	#model.summary()

	valLabels = []
	elevenEleven = []

	for file in sorted(os.listdir(samplePath)):
	        if file.startswith('test_label'):
	                valLabels.append(np.load(samplePath+file))
	        elif file.startswith('test_patch'):
	                elevenEleven.append(np.load(samplePath+file))

	valLabels = np.concatenate(valLabels,axis=0)
	elevenEleven = np.concatenate(elevenEleven,axis=0)

	# print(elevenEleven.shape)
	# print(valLabels.shape)

	# print(valLabels[0])

	Y_test = np.argmax(valLabels, axis=1)
	y_pred = model.predict_classes(elevenEleven)

	print('Patch dimension',dim,'report:')
	print(classification_report(Y_test, y_pred))
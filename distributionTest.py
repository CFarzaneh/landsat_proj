import matplotlib.pyplot as plt
import numpy as np
import os

directory = '8class_new60percent'
samplesDir = sorted(os.listdir(directory))

for folders in samplesDir:
	if folders.startswith('8classes'):
		print(folders)
		files = sorted(os.listdir(directory+'/'+folders))
		labelNums = []
		for file in files:
			if 'Y' in file:
				labels = np.load(directory+'/'+folders+'/'+file)
				labelNums = []
				for oneHot in labels:
					labelNums.append(int(np.argmax(oneHot)))
				print('Number of training samples:',len(labelNums))
				myDict = {k:labelNums.count(k) for k in set(labelNums)}
				print(myDict)
				plt.bar(myDict.keys(), myDict.values())
				plt.title('Training patch dimenstion: ' +folders.split("dim_",1)[1] + '     ' + str(len(labelNums)) + ' samples')
				plt.show()
				labelNums = []
			elif 'label' in file:
				labels = np.load(directory+'/'+folders+'/'+file)
				for oneHot in labels:
					labelNums.append(int(np.argmax(oneHot)))
		print('Number of test samples:',len(labelNums))
		myDict = {k:labelNums.count(k) for k in set(labelNums)}
		print(myDict)
		plt.bar(myDict.keys(), myDict.values())
		plt.title('Test patch dimenstion: ' +folders.split("dim_",1)[1] + '     ' + str(len(labelNums)) + ' samples')
		plt.show()
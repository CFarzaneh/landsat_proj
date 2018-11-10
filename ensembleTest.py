from keras.models import load_model
import numpy as np
import os
from tqdm import tqdm

samplePath = '/Users/cfarzaneh/Desktop/8classes_with_path_dim_3_HACKED/'

valLabels = []
elevenEleven = []
nineNine = []
sevenSeven = []
fiveFive = []
threeThree = []

for file in sorted(os.listdir(samplePath)):
        if file.startswith('test_label'):
                valLabels.append(np.load(samplePath+file))
        elif file.startswith('test_patch'):
                elevenEleven.append(np.load(samplePath+file))

valLabels = np.concatenate(valLabels,axis=0)
elevenEleven = np.concatenate(elevenEleven,axis=0)

for sample in elevenEleven:
	nineNine.append(sample[1:-1,1:-1])
	sevenSeven.append(sample[2:-2,2:-2])
	fiveFive.append(sample[3:-3,3:-3])
	threeThree.append(sample[4:-4,4:-4])

nineNine = np.stack(nineNine, axis=0)
sevenSeven = np.stack(sevenSeven, axis=0)
fiveFive = np.stack(fiveFive, axis=0)
threeThree = np.stack(threeThree, axis=0)

print("Loading models for ensemble")
models = []
i = 0
for model in tqdm(['model_3dim.h5','model_5dim.h5','model_7dim.h5','model_9dim.h5','model_11dim.h5']):
	model = load_model(model)
	models.append(model)
	if i == 0:
		score = model.evaluate(threeThree,valLabels,verbose=1)
	elif i == 1:
		score = model.evaluate(fiveFive,valLabels,verbose=1)
	elif i == 2:
		score = model.evaluate(sevenSeven,valLabels,verbose=1)
	elif i == 3:
		score = model.evaluate(nineNine,valLabels,verbose=1)
	elif i == 4:
		score = model.evaluate(elevenEleven,valLabels,verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	i+=1

print('Running ensemble')
predictions =  np.array(models[0].predict(threeThree))
predictions +=  np.array(models[1].predict(fiveFive))
predictions +=  np.array(models[2].predict(sevenSeven))
predictions +=  np.array(models[3].predict(nineNine))
predictions +=  np.array(models[4].predict(elevenEleven))

predictions = (predictions/len(models))

count = 0
for i in range(len(predictions)):
    correct_class = valLabels[i].argmax()
    guess = predictions[i].argmax()
    if guess == correct_class:
        count+=1

print('Ensemble accuracy: ',count/len(valLabels))
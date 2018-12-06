from keras.models import load_model
import numpy as np
import os

samplePath = '/Volumes/CamResearch/landsat_proj/8class_new60percent/8classes_with_path_dim_3/'
samplePathHacked = '/Users/cfarzaneh/Desktop/8classes_with_path_dim_3_HACKED/'

valLabels = []
valSamples = []

for file in sorted(os.listdir(samplePath)):
        if file.startswith('test_label'):
                valLabels.append(np.load(samplePath+file))
        elif file.startswith('test_patch'):
                valSamples.append(np.load(samplePath+file))

valLabels = np.concatenate(valLabels,axis=0)
valSamples = np.concatenate(valSamples,axis=0)


valLabelsPwned = []
valSamplesPwned = []

for file in sorted(os.listdir(samplePathHacked)):
        if file.startswith('test_label'):
                valLabelsPwned.append(np.load(samplePathHacked+file))
        elif file.startswith('test_patch'):
                valSamplesPwned.append(np.load(samplePathHacked+file))

valLabelsPwned = np.concatenate(valLabelsPwned,axis=0)
valSamplesPwned = np.concatenate(valSamplesPwned,axis=0)

print(valSamples[1])
print(valSamplesPwned[1])






# threeThree = []
# for sample in valSamples:
# 	threeThree.append(sample[4:-4,4:-4])
# valSamples = np.stack(threeThree, axis=0)

# model = load_model("model_3dim.h5")
# model.summary()

# print(valLabels.shape)
# print(valSamples.shape)

# score = model.evaluate(valSamples,valLabels,verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
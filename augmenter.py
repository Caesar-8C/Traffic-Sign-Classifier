import tensorflow as tf
import numpy as np
import pickle

train = pickle.load(open('trafsignsDataset/train.p', 'rb'))

augmented = []
labels = []

for i in range(train['features'].shape[0]):
	image = train['features'][i]

	augmented.append(tf.image.random_saturation(image, 3, 5))
	augmented.append(tf.image.random_brightness(image, 0.2))
	labels.append(train['labels'][i])
	labels.append(train['labels'][i])
	print(i, '/', train['features'].shape[0]-1)

augmented = np.array(augmented)
labels = np.array(labels)
train['features'] = np.append(train['features'], augmented, axis=0)
train['labels'] = np.append(train['labels'], labels)

pickle.dump(train, open('trafsignsDataset/train_augmented.p', 'wb'))
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import numpy as np
import pickle
from plotter import Plotter

train = pickle.load(open('trafsignsDataset/train_augmented.p', 'rb'))
valid = pickle.load(open('trafsignsDataset/valid.p', 'rb'))


train['features'] = ((train['features'].astype(np.int)-128)/128)
valid['features'] = ((valid['features'].astype(np.int)-128)/128)

train_ds = tf.data.Dataset.from_tensor_slices(
		(train['features'], train['labels'])).shuffle(100000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
		(valid['features'], valid['labels'])).batch(32)


class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = Conv2D(32, 5, activation='relu')
		self.pool1 = MaxPooling2D()
		self.conv2 = Conv2D(32, 3, activation='relu')
		self.pool2 = MaxPooling2D()
		self.flatten = Flatten()
		self.d1 = Dense(512, activation='relu')
		self.d2 = Dense(256, activation='relu')
		self.d3 = Dense(43)

	def call(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.flatten(x)
		x = self.d1(x)
		x = self.d2(x)
		return self.d3(x)


model = MyModel()
# model = tf.keras.models.load_model('models/model2')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images, training=True)
		loss = loss_object(labels, predictions)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
	predictions = model(images, training=False)
	t_loss = loss_object(labels, predictions)

	test_loss(t_loss)
	test_accuracy(labels, predictions)

EPOCHS = 30
plotter = Plotter(host='infiny.ddns.net', port='6008')

for epoch in range(EPOCHS):
	for fun in [train_loss, train_accuracy, test_loss, test_accuracy]: fun.reset_states()

	for images, labels in train_ds:
		train_step(images, labels)

	for test_images, test_labels in test_ds:
		test_step(test_images, test_labels)

	template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
	print(template.format(epoch,
						  train_loss.result(),
						  train_accuracy.result(),
						  test_loss.result(),
						  test_accuracy.result()))

	plotter.plot('namer', 'train', epoch, train_loss.result(), ytype='log')
	plotter.plot('namer', 'val', epoch, test_loss.result(), ytype='log')

model.save('models/model4')
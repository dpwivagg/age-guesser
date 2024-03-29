from create_model import create_cnn_model
import pretrained
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import optimizers
import tensorflow as tf

import numpy as np
import cv2
import pickle
import datetime # For naming files

model = pretrained.create_cnn_model()
model.summary()
model.load_weights('out/model_weightsMar-19-1658.h5')
print(len(model.layers))
for layer in model.layers[:32]:
    layer.trainable = False

rate = 0.001
print "Learning rate: %.4f" % rate
adam = optimizers.adam(lr=rate)
model.compile(loss='mean_squared_error',
	      optimizer=adam,
	      metrics=['mean_squared_error'])

batch_size = 64
print "Batch size: %i" % batch_size

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

faces = np.load('data/faces_train.npy')
ages = np.load('data/ages_train.npy')
np.round(ages)
faces = np.expand_dims(faces, axis=3)
#ages_cat = tf.keras.utils.to_categorical(ages, num_classes=101)
x_train, x_valid, y_train, y_valid = train_test_split(faces, ages, test_size=0.25, shuffle= True)
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch_size)

model.fit_generator(
        train_generator,
        steps_per_epoch=y_train.shape[0] // batch_size,
        epochs=60,
        validation_data=validation_generator,
        validation_steps=y_valid.shape[0] // batch_size,
	verbose=2)

filename = 'out/model_weights%s.h5' % datetime.datetime.today().strftime('%b-%d-%H%M')
model.save_weights(filename)
print("model saved in file %s" % filename) 

# img1 = np.array(img1).reshape((1, 200, 200, 3))
# prediction = model.predict_classes(img1)

score = model.evaluate_generator(validation_generator, steps=1, verbose=1)
print(score)
print(model.metrics_names)

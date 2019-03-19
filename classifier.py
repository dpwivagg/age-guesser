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
#model.load_weights('out/model_weightsMar-16-1606.h5')
rate = 0.005
print "Learning rate: %f" % rate
adam = optimizers.adam(lr=rate)
sgd = optimizers.sgd(lr=0.1)
model.compile(loss='sparse_categorical_crossentropy',
	      optimizer=sgd,
	      metrics=['accuracy', 'mean_squared_error'])

batch_size = 128
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

# labels = (train_generator.class_indices)
# label_map = dict((v,k) for k,v in labels.items())
# with open('out/labels.pkl', 'wb') as f:
#     pickle.dump(label_map, f, pickle.HIGHEST_PROTOCOL)
model.fit_generator(
        train_generator,
        steps_per_epoch=y_train.shape[0] // batch_size,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=y_valid.shape[0] // batch_size,
	verbose=2)

with open('out/loss-over-time.pkl', 'wb') as f:
    pickle.dump(model.history, f, pickle.HIGHEST_PROTOCOL)

model.save_weights('out/model_weights%s.h5' % datetime.datetime.today().strftime('%b-%d-%H%M'))

# img1 = cv2.imread('../Images/n02086910-papillon/n02086910_54.jpg')
# img1 = cv2.resize(img1, (200,200))
#
# img1 = np.array(img1).reshape((1, 200, 200, 3))
# prediction = model.predict_classes(img1)
#
# with open('out/labels.pkl', 'rb') as f:
#     label_map = pickle.load(f)
#
# print(label_map[prediction[0]])

score = model.evaluate_generator(validation_generator, steps=1, verbose=1)
print(score)
print(model.metrics_names)

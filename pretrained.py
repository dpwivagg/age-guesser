from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

def create_cnn_model(num_convolutions, num_hidden_layers, units_per_layer):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(16, (3, 3), input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(GlobalAveragePooling2D())

    #model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    for i in range(num_hidden_layers):
	model.add(Dense(units_per_layer))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

    model.add(Dense(101))
    model.add(Activation('softmax'))

    return model

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.utils import print_summary

def create_cnn_model():
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape=(48,48,1)))
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(2048, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(2048, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(101, (1, 1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('linear'))
#    model.add(Activation('softmax'))

    model.summary()
    return model

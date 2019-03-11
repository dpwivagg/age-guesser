from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (3, 3), input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(246))
    model.add(Activation('relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('linear'))

    return model

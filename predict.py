import pretrained
import numpy as np
from keras import optimizers
from keras.models import load_model

def predict(weights_path, faces_path, ages_path=None):
    # model = pretrained.create_cnn_model()
    # model.load_weights(weights_path)
    model = load_model(weights_path)
    adam = optimizers.adam(lr=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mean_squared_error'])
    faces = np.load(faces_path)
    faces = np.expand_dims(faces, axis=3)

    prediction = model.predict(faces)

    if ages_path:
        ages = np.load(ages_path)
        print(model.evaluate(faces, ages))
        print(model.metrics_names)

    return prediction.squeeze()
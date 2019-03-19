import pretrained
import numpy as np

def predict(weights_path, faces_path, ages_path=None):
    model = pretrained.create_cnn_model()
    model.load_weights(weights_path)
    faces = np.load(faces_path)
    faces = np.expand_dims(faces, axis=3)

    prediction = model.predict_classes(faces)

    if ages_path:
        ages = np.load(ages_path)
        print(model.evaluate(faces, ages))
        print(model.metrics_names)
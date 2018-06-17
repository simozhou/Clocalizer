import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf


def init():
    json_model = open("./model/cnn_arch.json")
    json_model_loaded = json_model.read()
    json_model.close()

    loaded_model = model_from_json(json_model_loaded)
    # load weights into new model
    loaded_model.load_weights("./model/cnn_weights.h5")

    print("Loaded Model from disk")

    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    graph = tf.get_default_graph()

    return loaded_model, graph

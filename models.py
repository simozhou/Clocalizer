import tensorflow as tf
import tensorboard as tb
"""
this module is just for collecting all models for finding the best one. A tensorflow model should look something like:

def my_model_fn(
   features,  This is batch_features from input_fn
   labels,    This is batch_labels from input_fn
   mode,      An instance of tf.estimator.ModeKeys
   params):   Additional configuration --> dict('feature_columns', 'hidden_layers', 'n_classes')

    every respectable model should contain at least:
    - an input layer
    - some hidden layers (cnn/lstm/whatevur)
    - some dense layer
    - an output layer with a softmax function for proper classification

    MODE ---> they are three in an estimator (train, evaluate, predict) and they need to be taken into account within
    the model.

    for train and evaluate: optimization by minimizing cross entropy (by adam or basic sgd) AND an EstimatorSpec
    instance

    for predict: just an EstimatorSpec instance (in prediction mode we don't expect to optimize furtherly our NN
    
    Let's try to include some Tensorboard code to have some beautiful graphics to display to some cute biologists!

"""


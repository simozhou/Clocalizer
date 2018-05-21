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


def cnn_pool_lstm(features, labels, mode, params):
    x = features['X']

    # 2 cnn layers on ReLU + pooling and an LSTM

    conv1 = tf.layers.conv2d(inputs=x, filters=50, kernel_size=3,
                             padding="same", activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=80, kernel_size=5,
                             padding="same", activation=tf.nn.relu)

    pool = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, )

    pool_flat = tf.layers.flatten(pool)

    lstm_f = tf.keras.layers.LSTM(units=[128, 15], input=pool_flat,
                                  kernel_initializer=tf.initializers.truncated_normal(mean=0.01), )
    lstm_b = tf.keras.layers.LSTM(units=[128, 15], input=lstm_f,
                                  kernel_initializer=tf.initializers.truncated_normal(mean=0.01), go_backwards=True)

    concat_lstm = tf.concat([lstm_f, lstm_b])

    dense_1 = tf.layers.dense(inputs=concat_lstm, activation=tf.nn.relu)

    logits = tf.layers.dense(dense_1, units=10)

    y_pred = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

    y_pred_cls = tf.arg_max(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:

        pass
#!usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Masking
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau

"""
this is where all models are built and trained. Keras Sequential is exploited, which will 
"""

# SIMPLE CNN
cnn = Sequential(name="Convolutional")

#
cnn.add(Conv2D(input_shape=(1000, 20, 1), filters=30, kernel_size=3, activation='relu'))
# cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters=30, kernel_size=5, activation='relu'))
# cnn.add(Dropout(0.2))
cnn.add(MaxPooling2D((2, 2), strides=2))

cnn.add(Flatten())

cnn.add(Dense(units=1024))
cnn.add(Dropout(0.4))
cnn.add(Dense(units=10, activation='softmax'))

# CNN WITH BIDIRECTIONAL LSTM
cnn_lstm = Sequential(name="Convolutional_lstm")

cnn_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=3, activation='relu', padding='same'),
                             input_shape=(None, 1000, 20, 1)))
# cnn_lstm.add(TimeDistributed(Dropout(0.2)))
cnn_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=5, activation='relu', padding='same')))
# cnn_lstm.add(TimeDistributed(Dropout(0.2)))
cnn_lstm.add(TimeDistributed(MaxPooling2D((2, 2), strides=2)))
cnn_lstm.add(TimeDistributed(Flatten()))
cnn_lstm.add(TimeDistributed(Masking(mask_value=0.0)))
cnn_lstm.add(Bidirectional(LSTM(100, recurrent_dropout=0.4)))

# cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(Dense(units=1024))
cnn_lstm.add(Dropout(0.4))
cnn_lstm.add(Dense(units=10, activation='softmax'))

# DOUBLE CNN WITH BIDIRECTIONAL LSTM
cnn2_lstm = Sequential(name="2-Convolutional_LSTM")

# cnn 1

cnn2_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=3, activation='relu', padding='same'),
                              input_shape=(None, 1000, 20, 1)))
# cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=5, activation='relu', padding='same')))
# cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(MaxPooling2D((2, 2), strides=2)))

# cnn2
cnn2_lstm.add(TimeDistributed(Conv2D(filters=60, kernel_size=5, padding='same', activation='relu')))
# cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(Conv2D(filters=60, kernel_size=7, padding='same', activation='relu')))
# cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(MaxPooling2D((2, 2), strides=2)))
cnn2_lstm.add(TimeDistributed(Flatten()))

# 2-dir lstm
cnn2_lstm.add(Bidirectional(LSTM(100, recurrent_dropout=0.4)))

cnn2_lstm.add(Dense(units=1024))
cnn2_lstm.add(Dropout(0.4))
cnn2_lstm.add(Dense(units=10, activation='softmax'))

models = dict(cnn=cnn, cnn_lstm=cnn_lstm, cnn2_lstm=cnn2_lstm)

# for command line interface and argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', help="pick a model to train [ cnn | cnn_lstm | cnn2_lstm ]")
parser.add_argument('-ep', '--epochs', help="Number of epochs you want to train your model", default=1000)
parser.add_argument('-lr', '--learning-rate', help="learning rate for Adam optimizer", default=1e-4)

args = parser.parse_args()

if args.model is None or args.model.lower() not in models.keys():
    parser.print_help()
    sys.stderr.write("Please specify training model!\n")
    sys.exit(1)

train_ds, test_ds = np.load("train.npz"), np.load("test.npz")

partition, x_train, y_train = train_ds["partition"], train_ds["X_train"], train_ds["y_train"]
x_test, y_test = test_ds['X_test'], test_ds['y_test']

l_rate, epochs = float(args.learning_rate), int(args.epochs)

decay = l_rate / epochs

adam = Adam(l_rate, decay=decay)

# compiling of the model

model = models[args.model]


# deprecated
def loss_fn(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, tf.int32)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


# deprecated
def accuracy_fn(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.argmax(y_pred, 1)
    correct_predictions = tf.equal(y_pred, y_true)
    return tf.reduce_mean(tf.cast(correct_predictions, "float"))


model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# to log the model structure

model.summary()

# callbacks

csv_logger = CSVLogger(f"training_{args.model}.csv", append=True)

early_stopper = EarlyStopping(min_delta=0.01, patience=10)

l_rate_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=5e-5)

# training phase with cross-validation at each step

for i in range(1, 5):
    # we partition the dataset so to have 4 groups for cross-validation

    x_part, y_part, x_val, y_val = x_train[np.where(partition != i)], y_train[np.where(partition != i)], x_train[
        np.where(partition == i)], y_train[np.where(partition == i)]

    # reshaping the dataset to fit cnn needs
    if args.model != "cnn":
        x_part, x_val = np.reshape(x_part, (x_part.shape[0], 1, x_part.shape[1], x_part.shape[2], 1)), \
                        np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1], x_val.shape[2], 1))
    else:
        x_part, x_val = np.reshape(x_part, x_part.shape + (1,)), np.reshape(x_val, x_val.shape + (1,))

    model.fit(x_part, y_part, batch_size=128, epochs=epochs // 4, validation_data=(x_val, y_val),
              callbacks=[csv_logger, early_stopper, l_rate_reducer])

# final evaluation

if args.model != "cnn":
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], 1))
else:
    x_test = np.reshape(x_test, x_test.shape + (1,))

scores, acc = model.evaluate(x_test, y_test, batch_size=20, verbose=0)

perc_scores = round(acc * 100, 3)

print(f'Accuracy: {perc_scores}%')

# saving model architecture and weights

mod_json = model.to_json()

with open(f"{args.model}_arch.json", 'w') as json_file:
    json_file.write(mod_json)

model.save_weights(f'{args.model}_weights.h5')

# switch off the instance

os.system('sudo shutdown')

#!usr/bin/env python3

import argparse
import os
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, TimeDistributed
from keras.optimizers import Adam

"""
this is where all models are built and trained. Keras Sequential is exploited, which will 
"""

# SIMPLE CNN
cnn = Sequential(name="Convolutional")
#
cnn.add(Conv2D(input_shape=(1000, 20, 1), filters=30, kernel_size=3, activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters=30, kernel_size=5, activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(MaxPooling2D((2, 2)))

cnn.add(Flatten())

cnn.add(Dense(units=1024))
cnn.add(Dropout(0.2))
cnn.add(Dense(units=10, activation='softmax'))

# CNN WITH BIIRECTIONAL LSTM
cnn_lstm = Sequential(name="Convolutional_lstm")

cnn_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=3, activation='relu', ), input_shape=(128, 1000, 20, 1)))
cnn_lstm.add(TimeDistributed(Dropout(0.2)))
cnn_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=5, activation='relu')))
cnn_lstm.add(TimeDistributed(Dropout(0.2)))
cnn_lstm.add(TimeDistributed(MaxPooling2D((2, 2))))
cnn_lstm.add(TimeDistributed(Flatten()))
cnn_lstm.add(Bidirectional(LSTM(100)))

cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(Dense(units=1024))
cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(Dense(units=10, activation='softmax'))

# DOUBLE CNN WITH BIDIRECTIONAL LSTM
cnn2_lstm = Sequential(name="2-Convolutional_LSTM")

# cnn 1
cnn2_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=3, activation='relu'), input_shape=(128, 1000, 20, 1)))
cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(Conv2D(filters=30, kernel_size=5, activation='relu')))
cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(MaxPooling2D((2, 2))))

# cnn2
cnn2_lstm.add(TimeDistributed(Conv2D(filters=60, kernel_size=5, padding='same', activation='relu')))
cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(Conv2D(filters=60, kernel_size=7, padding='same', activation='relu')))
cnn2_lstm.add(TimeDistributed(Dropout(0.2)))
cnn2_lstm.add(TimeDistributed(MaxPooling2D((2, 2))))
cnn2_lstm.add(TimeDistributed(Flatten()))

# 2-dir lstm
cnn2_lstm.add(Bidirectional(LSTM(100)))

cnn2_lstm.add(Dense(units=1024))
cnn2_lstm.add(Dropout(0.2))
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

l_rate, epochs = int(args.learning_rate), int(args.epochs)

decay = l_rate / epochs

adam = Adam(l_rate, decay=decay)

# compiling of the model

model = models[args.model]

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# to log the model structure

model.summary()

# training phase with cross-validation at each step

for i in range(1, 5):
    # we partition the dataset so to have 4 groups for cross-validation

    x_part, y_part, x_val, y_val = x_train[np.where(partition != i)], y_train[np.where(partition != i)], \
                                   x_train[np.where(partition == i)], y_train[np.where(partition == i)]

    # reshaping the dataset to fit cnn needs

    x_part, x_val = np.reshape(x_part, x_part.shape + (1,)), np.reshape(x_val, x_val.shape + (1,))

    model.fit(x_part, y_part, batch_size=128, epochs=epochs, validation_data=(x_val, y_val))

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=0)

perc_scores = round(scores[1] * 100, 2)

print(f'Accuracy: {perc_scores}%')

# saving model architecture and weights

mod_json = model.to_json()

with open(f"{args.model}_arch.json", 'w') as json_file:
    json_file.write(mod_json)

model.save_weights(f'{args.model}_weights.h5')

# switch off the instance

os.system('sudo shutdown')
import numpy as np
import tensorflow as tf
import models as mo


# loading datasets and phreshing folds for evaluations and testing

tf.logging.set_verbosity('INFO')

train_ds, test_ds = np.load("train.npz"), np.load("test.npz")

partition, x_train, y_train = train_ds["partition"], train_ds["X_train"], train_ds["y_train"]
x_test, y_test = test_ds['X_test'], test_ds['y_test']

print("imports done")

feature_columns = [tf.feature_column.numeric_column("X", shape=(1000, 20))]

num_hidden_units = [1024, 512, 256, 128]

# model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
#                                    n_classes=10, model_dir="./checkpoints_FFN/", hidden_units=num_hidden_units)

model = tf.estimator.Estimator(model_fn=mo.cnn_pool, params={'learning_rate': 1e-4, 'n_classes': 10}, model_dir='./checkpoints_lstm1/')

print("Model built")

for i in range(1, 5):
    x_part, y_part, x_val, y_val = x_train[np.where(partition != i)], y_train[np.where(partition != i)], \
                                   x_train[np.where(partition == i)], y_train[np.where(partition == i)]

    print("partitions done!")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'X': x_part},
                                                        y=y_part,
                                                        num_epochs=1000,
                                                        batch_size=128,
                                                        shuffle=False)

    print("train input function built!")

    model.train(input_fn=train_input_fn, steps=1000)

    print(f"CV step #{i} completed")

final_test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'X': x_test},
                                                         y=y_test,
                                                         num_epochs=20,
                                                         batch_size=128,
                                                         shuffle=False)

model.evaluate(input_fn=final_test_input_fn)

# accuracy = 0.6468619, average_loss = 2.4478252, global_step = 4000, loss = 312.85037
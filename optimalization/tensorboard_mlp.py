import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import datetime, os

from tensorflow.python.ops.gen_nn_ops import lrn, softmax

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

def build_model(activation):
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation=activation),
        layers.Dense(64, activation=activation),
        layers.Dense(10, activation='softmax')
        ])
    return model

activations = ['relu', 'tanh', 'sigmoid']
lr = 0.00025
epochs = 30
batch_size = 128

log_root = 'logs/wariant14'
os.makedirs(log_root, exist_ok=True)

for act in activations:
    model = build_model(act)
    model.compile(optimizer=Adam(learning_rate=lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

    log_dir = f"{log_root}/{act}_lr{lr}_{datetime.datetime.now().strftime('%d%m%Y-%H%M%S')}"
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print("Train model with activation", act, "logdir:", log_dir)
    model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[tb]
            )

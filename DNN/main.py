import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris



def run_iris_dense():
    iris = load_iris()

    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137, stratify=y)

    model = models.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(32, activation='hard_sigmoid'),
        layers.Dense(16, activation='hard_sigmoid'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=8)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, Test acc: {acc:.4f}")

def run_MNIST_conv2D():
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype('float32') / 255.0
    x_test = x_test[..., np.newaxis].astype('float32') / 255.0

    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, (2,2), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (2,2), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=8)

    loss, acc = model.evaluate(x_test, y_test)

    print(f"Test loss: {loss:.4f}, Test acc: {acc:.4f}")


def transformer_blocK(input_shape, initializer):
    inp = layers.Input(shape=input_shape)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, kernel_initializer=initializer)(inp, inp)
    attn = layers.LayerNormalization(epsilon=1e-6)(attn + inp)
    ffn = layers.Dense(256, activation='relu', kernel_initializer=initializer)(attn)
    ffn = layers.Dense(input_shape[-1], kernel_initializer=initializer)(ffn)
    out = layers.LayerNormalization(epsilon=1e-6)(ffn + attn)
    return models.Model(inputs=inp, outputs=out)

def run_transformer_weight():

    X = np.random.rand(1000, 10, 64).astype('float32')
    Y = np.random.rand(1000, 10, 64).astype('float32')

    inits = {
            'glorot_uniform': initializers.GlorotUniform(),
            'he_normal': initializers.HeNormal(),
            'random_normal': initializers.RandomNormal(stddev=0.5)
            }
    
    results = {}

    for name, init in inits.items():
        print("Testing:", name)
        block = transformer_blocK((10, 64), initializer=init)
        block.compile(optimizer='adam', loss='mse')
        h = block.fit(X, Y, epochs=5, batch_size=32, verbose=1)
        loss = h.history['loss'][-1]
        print(f" final loss: {loss:.6f}")
        results[name] = loss

    print("summary:", results)



if __name__ == '__main__':
    run_iris_dense()
    print('----------')
    run_MNIST_conv2D()
    print("----------")
    run_transformer_weight()


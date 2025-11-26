import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
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

if __name__ == '__main__':
    run_iris_dense()

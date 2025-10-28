# f(x) = x^2 - e^-x, x € [1;10]

from os import wait
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Found {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(f"⚠️ GPU initialization error: {e}")
else:
    print("⚙️ No GPU available, running on CPU.")

x_train = np.linspace(1, 10, 1000)
#print(x_train)

y_train = x_train**2 - np.exp(-x_train)

#print(y_train)

x_mean, x_std = np.mean(x_train), np.std(x_train)
x_norm = (x_train - x_mean) / x_std

y_mean, y_std = np.mean(y_train), np.std(y_train)
y_norm = (y_train - y_mean) / y_std




model = keras.Sequential([
    keras.Input(shape=(1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(43, activation='relu'),
    layers.Dense(1)
])

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x_norm, y_norm, epochs=300, verbose=0)

loss = model.evaluate(x_norm, y_norm, verbose=1)
print(f'MSE (znormalizowany) {loss}')

y_pred_norm = model.predict(x_norm)
y_pred = y_pred_norm * y_std + y_mean


if loss < 0.01:
    print("Błąd mniejszy niz 0.01")
else:
    print("Failed")

plt.figure(figsize=(8, 6))
plt.plot(x_train, y_train, label='Rzeczywista funckja', linewidth=2)
plt.plot(x_train, y_pred, label='Predykcja NN', linestyle='--')
plt.legend()
plt.title("funckja x^2 - e^-x, x € [1;10]")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()




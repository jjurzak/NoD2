import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
from pandas.core.common import random_state
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM 
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from tensorflow.python.checkpoint.checkpoint import metrics
from tensorflow.python.eager.context import anonymous_name


def generate_forecasting_data(n_samples=1000, freq=0.05, noise=0.1):
    x = np.arange(n_samples)
    y = np.sin(2 * np.pi * freq * x) + np.random.normal(0, noise, size=n_samples)
    return y.reshape(-1, 1)

data = generate_forecasting_data(1000)
scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)

#print(data_scaled)

def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10 
X, y = create_dataset(data_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential([
        layers.LSTM(100, input_shape=(look_back, 1)),
        layers.Dense(1)
        ])

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=1, verbose=1)




predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y.reshape(-1, 1))

plt.plot(real, label='Rzeczywiste')
plt.plot(predicted, label='Prognozowane')
plt.legend()
plt.title('Prognozowane wartosci')
plt.show()


def generate_anomaly_data(n_samples=1000, timesteps=10, anomaly_rate=0.1):
    X = np.random.normal(0, 1, (n_samples, timesteps))
    y = np.zeros(n_samples)
    n_anomalies = int(anomaly_rate * n_samples)
    anomalies = np.random.choice(n_samples, n_anomalies, replace=False)
    X[anomalies] += np.random.normal(5, 1, (n_anomalies, timesteps))
    y[anomalies] = 1 
    return X.reshape(n_samples, timesteps, 1), y




X, y = generate_anomaly_data(1000)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2137, stratify=y
        )

X_train_2d = X_train.reshape(-1, X_train.shape[-1])   # (n_samples * timesteps, 1)
X_train_scaled_2d = scaler.fit_transform(X_train_2d)  # scaler widzi tablicę 2D
X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)

X_test_2d = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled_2d = scaler.transform(X_test_2d)
X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

model = Sequential([
    layers.LSTM(100, input_shape=(X.shape[1], 1)),
    layers.Dense(1, activation='sigmoid')
    ])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=25, batch_size=32, validation_data=(X_test_scaled, y_test),verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Dokladnosc: {accuracy:.2f}')

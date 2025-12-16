import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM 
from sklearn.preprocessing import MinMaxScaler, scale


def generate_forecasting_data(n_samples=1000, freq=0.05, noise=0.1):
    x = np.arange(n_samples)
    y = np.sin(2 * np.pi * freq * x) + np.random.normal(0, noise, size=n_samples)
    return y.reshape(-1, 1)

def genrate_anomaly_data(n_samples=1000, timesteps=10, anomaly_rate=0.1)
    X = np.random.normal(0, 1, (n_samples, timesteps))
    y = np.zeros(n_samples)
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
model.fit(X, y, epochs=35, batch_size=1, verbose=1)

predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y.reshape(-1, 1))

plt.plot(real, label='Rzeczywiste')
plt.plot(predicted, label='Prognozowane')
plt.legend()
plt.title('Prognozowane wartosci')
plt.show()




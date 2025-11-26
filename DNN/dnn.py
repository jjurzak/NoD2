import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def run_iris_dense():
    iris = load_iris()
    
    x, y = iris.data, iris.target
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2137, stratify=y)

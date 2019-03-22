import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import sys

from sklearn.preprocessing import StandardScaler

fashion_mnist = tf.keras.datasets.fashion_mnist

# Get a subpart of the dataset
(images, targets), (_, _) = fashion_mnist.load_data()

# Get a subpart
# images = images[:10000]
# targets = targets[:10000]

# Before Normalization
# print("before normalization")
# print("Average", images.mean())
# print("std", images.std())

# Normalization of the dataset to allow to modify weights quickly
images = images.reshape(-1, 784)
images = images.astype(float)
scaler = StandardScaler() # z=(x-average)/std
images = scaler.fit_transform(images)

# print(images.shape)
# print(targets.shape)

# After Normalization
# print("After Normalization")
# print("Average", images.mean())
# print("std", images.std())

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot an image
# plt.imshow(images[10].reshape(28,28), cmap='binary')
# plt.title(class_names[targets[10]])
# plt.show()

# Flatten
model = tf.keras.models.Sequential()


# Add the layers
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model_output = model.predict(images[0:1])
print(model_output, targets[0:1])

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(images, targets, epochs=10)

# New prediction
model_output = model.predict(images[0:1])
print(model_output, targets[0:1])

# loss curve
loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

plt.plot(loss_curve)
plt.title("Loss")
plt.show()

plt.plot(acc_curve)
plt.title("Accuracy")
plt.show()
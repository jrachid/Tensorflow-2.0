import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_,_) = fashion_mnist.load_data()

images = images.reshape(-1,784)
images = images.astype(float)
scaler = StandardScaler()
images = scaler.fit_transform(images)

images_train, images_test, targets_train, targets_test = train_test_split(images, targets, test_size=0.2, random_state=1)
print(images_train.shape, targets_train.shape)
print(images_test.shape, targets_test.shape)

# Add Layers
# Flatten
model = tf.keras.models.Sequential()

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

# Train the model only on images_train and targets_train
history = model.fit(images_train, targets_train, epochs=100, validation_split=0.2)

# loss and val_loss curves
loss = history.history["loss"]
val_loss = history.history["val_loss"]

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

plt.plot(loss, label="Train")
plt.plot(val_loss, label="Val")
plt.legend(loc="upper left")
plt.title("LOSS")
plt.show()

plt.plot(acc, label="Train")
plt.plot(val_acc, label="Val")
plt.legend(loc="upper left")
plt.title("Accuracy")
plt.show()

# Save the Model
model.save('Models\\simple_model.h5')
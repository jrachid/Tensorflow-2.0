import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import sys

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot an image
# plt.imshow(images[11], cmap="binary")
# plt.title(class_names[targets[11]])
# plt.show()

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28]))

# print("Shape of the image", images[0:1].shape)
# model_output = model.predict(images[0:1])
# print("Shape of the image after the flatten", model_output.shape)

# Add the layers
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# First prediction
model_output = model.predict(images[0:1])
print(model_output, targets[0:1])

model.summary()

# Compile the model
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer='sgd', # stocastic gradient descent
    metrics=['accuracy']
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
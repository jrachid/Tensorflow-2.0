import tensorflow as tf 
import numpy as np 

# Eager Mode
a =np.array([1., 2.])
b =np.array([2., 5.])

print(tf.add(a, b))

# Graph Model : AutoGraph
@tf.function # Allow to transform Eager Mode to Graph Mode
def add_fc(a,b):
    return tf.add(a,b)

print(add_fc(a, b))

# Create the Graph
# def add_fc(a,b):
#     return tf.add(a, b)

# print(tf.autograph.to_code(add_fc))

# Graph mode and eager mode with a Keras Model
# Flatten
model = tf.keras.models.Sequential()
# Add layers
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# 2 types of outputs
# Array output
model_output = model.predict(np.zeros((1,30)))
print(model_output)

# Tensor output
model_output = model(np.zeros((1,30)))
print(model_output)

# We can execute the model in graph mode by using the @tf.function decorator
@tf.function
def predict(x):
    return model(x)

model_output = predict(np.zeros((1, 30))) 
print(model_output)
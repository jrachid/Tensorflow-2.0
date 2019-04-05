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

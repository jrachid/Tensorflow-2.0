import tensorflow as tf 
import numpy as np 

# Create a Model
model = tf.keras.models.Sequential()
# Add layers
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(2, activation="softmax"))

# The loss method
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# The optimizer
optimizer = tf.keras.optimizers.Adam()

# Accumulateur : This metrics is used to track the progress of the training loss during the training
train_loss = tf.keras.metrics.Mean(name='train_loss')

# Create a method which used autograph to train the model
@tf.function
def train_step(image, targets):
    with tf.GradientTape() as tape:
        # Make a prediction
        predictions = model(image)
        # Get the error/loss using the loss_object previously defined
        loss = loss_object(targets, predictions)
    # Compute the gradient which respect to the loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # Change the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # The metrics are accumulate over time. Don't need to average it
    train_loss(loss)

for epoch in range (0,10):
    for _ in range(0, 10):
        # Create fake inputs with two classes
        # Create fake inputs
        inputs = np.zeros((2,30))
        inputs[0] -= 1
        inputs[1] = 1
        # Create fake targets
        targets = np.zeros((2,1))
        targets[0] = 0
        targets[1] = 1
        # Train the model
        train_step(inputs, targets)
    print("Loss: %s" % train_loss.result())
    train_loss.reset_states()

# try:
#     input_ = np.zeros((1,30)) + 1
#     model.predict(input_)
# except Exception as e:
#     print("error ===>", e)
    
input_ = np.zeros((1,30)) + 1
print(model(input_))

input_ = np.zeros((1,30)) - 1
print(model(input_))
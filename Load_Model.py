import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load data: Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_,_) = fashion_mnist.load_data()

images = images.reshape(-1,784)
images = images.astype(float)
scaler = StandardScaler()
images = scaler.fit_transform(images)

images_train, images_test, targets_train, targets_test = train_test_split(images, targets, test_size=0.2, random_state=1)

# Load Model
loaded_model = tf.keras.models.load_model('Models\\simple_model.h5')
loss, acc = loaded_model.evaluate(images_test, targets_test)
print("Loss: ", loss)
print("Accuracy: ", acc)

# Make a prediction

np.set_printoptions(formatter={'float': '{: 0.9f}'.format})
print(loaded_model.predict(images_test[0:1].astype(float)), class_names[targets_test[1]], targets_test[0:1])

# Plot an image
plt.imshow(images_test[0:1].reshape(28,28), cmap='binary')
plt.show()


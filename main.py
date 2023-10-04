

print("------ Single Layer Perceptron in TensorFlow ------")

# Step1: Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline



# Step 2: Now load the dataset using “Keras” from the imported version of tensor flow.
(x_train, y_train), \
(x_test, y_test) = keras.datasets.mnist.load_data()
print(y_train)



# Step 3: Now display the shape and image of the single image in the dataset.
# The image size contains a 28*28 matrix and length of the training set is 60,000 and the testing set is 10,000.
len(x_train)
len(x_test)
x_train[0].shape
plt.matshow(x_train[0])



# Step 4: Now normalize the dataset in order to compute the calculations in a fast and accurate manner.
# Normalizing the dataset
x_train = x_train / 255
x_test = x_test / 255



# Flatting the dataset in order to compute for model building
x_train_flatten = x_train.reshape(len(x_train), 28 * 28)
x_test_flatten = x_test.reshape(len(x_test), 28 * 28)



# Step 5: Building a neural network with single-layer perception.
# Here we can observe as the model is a single-layer perceptron that only contains one input layer
# and one output layer there is no presence of the hidden layers.
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),
                       activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train_flatten, y_train, epochs=5)



# Step 6: Output the accuracy of the model on the testing data.
print("\nOutput the accuracy of the model on the testing data:")
model.evaluate(x_test_flatten, y_test)

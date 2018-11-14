# -*- coding: utf-8 -*-
"""
Sources:
https://www.tensorflow.org/tutorials/keras/basic_classification
https://elitedatascience.com/keras-tutorial-deep-learning-in-python

To run the program, specify appropriate values for SOURCE and NETWORK,
this will lead to 4 different configurations of the program
"""

SOURCE = 'digits' # fashion|digits -- type of source data
NETWORK = 'simple' # simple|conv -- type of neural architecture

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

print(tf.__version__)

if SOURCE=='fashion':
    dset = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = dset.load_data()
elif SOURCE=='digits':
    dset = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = dset.load_data()
print(train_images.shape)
if SOURCE=='fashion':
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
elif SOURCE=='digits':
    class_names = ['0-zero', '1-one', '2-two', '3-three', '4-four', 
               '5-five', '6-six', '7-seven', '8-eight', '9-nine']

test_images_orig = test_images
if NETWORK=='conv':
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_labels_cat = np_utils.to_categorical(train_labels, 10)
test_labels_cat = np_utils.to_categorical(test_labels, 10)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images / 255
test_images = test_images / 255

model = keras.Sequential()
if NETWORK=='simple':
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.softmax))
elif NETWORK=='conv':
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28, 1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if NETWORK=='simple':
    model.fit(train_images, train_labels_cat, epochs=5)
elif NETWORK=='conv':
    model.fit(train_images, train_labels_cat, batch_size=32, epochs=1, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
print('Test accuracy:', test_acc)
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print(test_labels[0])

# Plot one prediction
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images_orig)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images_orig)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
  
  
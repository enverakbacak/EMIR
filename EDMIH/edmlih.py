#coding=utf-8

import cv2
import glob
import numpy as np
import sys,os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
import glob
import os, os.path
import re
import scipy.io
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
import keras
from keras.models import Sequential,Input,Model,InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.applications import InceptionV3
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images
from keras.utils.vis_utils import plot_model

Y = pd.read_csv(r'/home/ubuntu/keras/enver/edmlih/Y.csv') # labels, one hot encoded
Y.shape
print(Y.shape[:])


image_size=224
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


X = np.load(open('preprocessed_X.npy'))
X.shape
print(X.shape)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1 ,random_state=43)


batch_size = 32
epochs = 5
hash_bits = 64

#visible = Input(shape=(7,7,512)) 
visible = Input(shape = base_model.output_shape[1:])
Flatten = Flatten()(visible)
Dense_1 = Dense(1024)(Flatten)
#batchNorm = BatchNormalization()(Dense_1)
Dense_2 = Dense(hash_bits ,activation='tanh')(Dense_1)
Dense_3 = Dense(5, activation='sigmoid')(Dense_2)
model = Model(input = visible, output=Dense_3)
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance

import keras.backend as K
# e = 0.5
def c_loss(noise_1, noise_2):
    def loss(y_true, y_pred):
        #return ( (K.binary_crossentropy(y_true, y_pred)) + (K.sum((noise_1 - noise_2)**2) ) * (1/hash_bits)   )
        return ( (K.binary_crossentropy(y_true, y_pred)) + (K.sum(K.binary_crossentropy(noise_1, noise_2) )) * (1/hash_bits)   )
 
    return loss



from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss = c_loss(noise_1 = tf.to_float(Dense_2 > 0.5 ), noise_2 = Dense_2 ),  optimizer=sgd, metrics=['accuracy'])
history = model.fit(X_train, Y_train, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_valid, Y_valid) )
#history = model.fit(X, Y, validation_split=0.1, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1 )

model_json = model.to_json()
with open("models/emqir_64_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/emqir_64_weights.h5")


params = {'legend.fontsize': 36,
          'legend.handlelength': 4,}
plt.rcParams.update(params)

matplotlib.rcParams.update({'font.size': 36})
plt.plot(history.history['acc'] , linewidth=5, color="green")
plt.plot(history.history['val_acc'], linestyle='--',  linewidth=5, color="red")
#plt.title('model accuracy' , fontsize=32)
plt.ylabel('Accuracy' , fontsize=40)
plt.xlabel('The number of epochs' , fontsize=36)
plt.legend( ['train', 'validation'], loc='best')
plt.show()
# summarize history for loss
matplotlib.rcParams.update({'font.size': 36})
plt.plot(history.history['loss'], linewidth=5, color="green")
plt.plot(history.history['val_loss'], linestyle='--', linewidth=5, color="red")
#plt.title('model loss' , fontsize=32)
plt.ylabel('Loss' , fontsize=40)
plt.xlabel('The number of epochs' , fontsize=36)
plt.legend( ['train', 'validation'], loc='best')
plt.show()


score = model.evaluate(X_valid, Y_valid, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(X_train, Y_train)
print(model.metrics_names)
print(score)

score = model.evaluate(X_valid, Y_valid)
print(model.metrics_names)
print(score)

score = model.evaluate(X, Y)
print(model.metrics_names)
print(score)


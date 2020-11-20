import cv2
import keras
import numpy as np
from keras.applications import InceptionV3
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images




data = pd.read_csv('data.csv')     # reading the csv file

X = [ ]     # creating an empty array
for myFile in data.Image_ID:
    image = plt.imread('/home/ubuntu/caffe/data/lamda/train/' + myFile) #   /home/ubuntu/caffe/data/lamda_2/lamdaPics/*.jpg is  alrready 256x256
    X.append (image)
X = np.array(X)    # converting list to array


image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)


from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')      # preprocessing the input data


image_size=224
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


batch_size=64
X = base_model.predict(X, batch_size=batch_size, verbose=0, steps=None)
np.save(open('preprocessed_X.npy', 'w'), X)

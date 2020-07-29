import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import os
import cv2
from tqdm import tqdm
import matplotlib as mlp
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Import train, test set

TrainImagePaths = []
for dirname, _, filenames in os.walk(dir1):
    for filename in filenames:
        if (filename[-3:] == 'jpg'):
            TrainImagePaths.append(os.path.join(dirname, filename))

ValImagePaths = []
for dirname, _, filenames in os.walk(dir2):
    for filename in filenames:
        if (filename[-3:] == 'jpg'):
            ValImagePaths.append(os.path.join(dirname, filename))
            
#Promvx

imgSize = 64
X_train = []
Y_train = []
for imagePath in tqdm(TrainImagePaths):
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgSize, imgSize))

    X_train.append(image)
    Y_train.append(int(label))
    
X_train = np.array(X_train).astype('float16')/255
Y_train = to_categorical(Y_train)

from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.utils import to_categorical
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")
import createTree
import random
import pdb

rows, cols,channels = 32,32,3

input_shape = (rows, cols, channels)


inp = Input(shape=input_shape)

conv1_h = Conv2D(32,(2,2),padding='same',input_shape=input_shape)
conv1_a = BatchNormalization()
conv1_b = Activation('relu')
conv1_c = Dropout(0.3)
conv1_d = Conv2D(32,(4,4),padding='same')
conv1_e = BatchNormalization()
conv1_f = Activation('relu')
conv1_g = MaxPooling2D(pool_size=(2,2),strides=2)

conv1 = conv1_g(conv1_f(conv1_e(conv1_d(conv1_c(conv1_b(conv1_a(conv1_h(inp))))))))

conv2_h = Conv2D(64,(4,4),padding='same')
conv2_a = BatchNormalization()
conv2_b = Activation('relu')
conv2_c = Dropout(0.4)
conv2_d = Conv2D(64,(2,2),padding='same')
conv2_e = BatchNormalization()
conv2_f = Activation('relu')
conv2_g = MaxPooling2D(pool_size=(2,2),strides=2)

conv2 = conv2_g(conv2_f(conv2_e(conv2_d(conv2_c(conv2_b(conv2_a(conv2_h(conv1))))))))

conv3_h = Conv2D(128,(2,2),padding='same')
conv3_a = BatchNormalization()
conv3_b = Activation('relu')
conv3_c = Dropout(0.25)
conv3_d = Conv2D(128,(3,3),padding='same')
conv3_e = BatchNormalization()
conv3_f = Activation('relu')
conv3_g = MaxPooling2D(pool_size=(2,2),strides=2)
conv3_i = Flatten()

conv3 = conv3_i(conv3_g(conv3_f(conv3_e(conv3_d(conv3_c(conv3_b(conv3_a(conv3_h(conv2)))))))))

model = Model(inputs=inp,outputs=conv3)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=.001), metrics = ['accuracy'])

BATCH_SIZE = 32
NUM_CLASSES = 100
EPOCHS = 200


(x_train,y_train),(x_test,y_test) = cifar100.load_data()
x_train, x_val, y_train, y_val = train_test_split(
	x_train,y_train,test_size=.1,random_state=2345432)



x_train = x_train.reshape(x_train.shape[0], rows, cols, channels)
x_test = x_test.reshape(x_test.shape[0], rows, cols, channels)
x_val = x_val.reshape(x_val.shape[0],rows,cols,channels)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /=255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)


datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train,batch_size=BATCH_SIZE),
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_val, y_val))


model.save('BURTA.h5')




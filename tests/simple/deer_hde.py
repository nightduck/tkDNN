import os
import errno

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU, Reshape, Lambda, Conv1D, Conv2D, DepthwiseConv2D
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import struct
from keras.models import Sequential, Model

input_shape = (3, 192, 192)

def bin_write(f, data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def create_model():

    hde = keras.Sequential()
    hde.add(keras.Input(shape=input_shape, name="data"))
    hde.add(Conv2D(32, (3,3), strides=(2,2), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(1,1), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(2,2), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(1,1), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(2,2), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(1,1), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(2,2), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(1,1), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(2,2), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(1,1), padding="same", data_format="channels_first"))
    hde.add(DepthwiseConv2D((3,3), strides=(2,2), padding="same", data_format="channels_first"))
    hde.add(Conv2D(32, (3,3), strides=(1,1), padding="same", data_format="channels_first"))
    hde.add(Flatten())
    hde.add(Dense(2, activation="sigmoid", name="out"))
    hde.summary()

    return hde

if __name__ == '__main__':
    print ("DATA FORMAT: ", keras.backend.image_data_format())

    model = create_model()
    model.save("net.hdf5")

    np.random.seed(2)
    x = np.random.rand(1,3,192,192)
    print(x.shape)
    r = model.predict(x, batch_size=1)

    r = np.array([r])
    x =  x.transpose(0, 3, 1, 2)
    #r =  r.transpose(0, 3, 1, 2)
    print("in: ", np.shape(x))
    print("out: ", np.shape(r))
    #print("output: ", r.tolist())

    if not os.path.exists('test_deer_hde'):
        os.makedirs('test_deer_hde')
    if not os.path.exists('test_deer_hde/debug'):
        os.makedirs('test_deer_hde/debug')
    if not os.path.exists('test_deer_hde/layers'):
        os.makedirs('test_deer_hde/layers')

    x = np.array(x.flatten(), dtype=np.float32)
    f = open("test_deer_hde/debug/input.bin", mode='wb')
    bin_write(f, x)

    r = np.array(r.flatten(), dtype=np.float32)
    f = open("test_deer_hde/debug/output.bin", mode='wb')
    bin_write(f, r)

    for layer in model.layers:
        data = layer.get_weights()
        if (len(data) != 0):
            weights = data[0].flatten()
            biases = data[1].flatten()
            f = open("test_deer_hde/layers/" + layer.name + ".bin", mode='wb')

            bin_write(f, np.concatenate([d.flatten() for d in data]))
            f.close()

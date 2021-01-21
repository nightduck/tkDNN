import os
import errno
import datetime

import keras
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import struct
from keras.models import Sequential, Model

input_shape = (192, 192, 3)

def bin_write(f, data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def create_model():
    x = keras.applications.MobileNet(input_shape, 1, include_top=False, weights="imagenet")
    x.layers[0]._name = "data"
    output = x.output
    output = layers.Conv2D(32, (3,3), strides=(2,2), use_bias=False, padding='valid')(output)
    output = layers.Flatten()(output)
    output = layers.Dense(64, activation="sigmoid")(output)
    output = layers.Dense(2, activation="sigmoid", name="out")(output)
    hde = keras.models.Model(x.input, output)
    hde.load_weights("hde_deer.hdf5")
    hde.summary()

    return hde

def create_compact_model():
    hde = keras.Sequential()
    hde.add(keras.Input(shape=input_shape, name="data"))
    hde.add(layers.Conv2D(32, (3,3), strides=(2,2), padding="same", name="conv2d_1"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(1,1), padding="same", name="conv_dw_2_dwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_2_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_2_dwrelu"))
    hde.add(layers.Conv2D(64, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_2_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_2_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_2_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(2,2), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_3_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_3_dwrelu"))
    hde.add(layers.Conv2D(128, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_3_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_3_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_3_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(1,1), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_4_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_4_dwrelu"))
    hde.add(layers.Conv2D(128, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_4_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_4_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_4_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(2,2), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_5_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_5_dwrelu"))
    hde.add(layers.Conv2D(256, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_5_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_5_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_5_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(1,1), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_6_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_6_dwrelu"))
    hde.add(layers.Conv2D(256, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_6_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_6_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_6_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(2,2), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_7_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_7_dwrelu"))
    hde.add(layers.Conv2D(512, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_7_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_7_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_7_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(1,1), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_8_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_8_dwrelu"))
    hde.add(layers.Conv2D(512, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_8_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_8_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_8_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(2,2), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_9_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_9_dwrelu"))
    hde.add(layers.Conv2D(1024, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_9_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_9_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_9_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(1,1), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_10_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_10_dwrelu"))
    hde.add(layers.Conv2D(1024, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_10_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_10_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_10_pwrelu"))

    hde.add(layers.DepthwiseConv2D((3,3), strides=(2,2), padding="same"))
    hde.add(layers.BatchNormalization(name="conv_dw_11_dwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_11_dwrelu"))
    hde.add(layers.Conv2D(2048, (1,1), padding="same", use_bias=False, strides=(1,1), name="conv_dw_111_pwconv2d"))
    hde.add(layers.BatchNormalization(name="conv_dw_11_pwbn"))
    hde.add(layers.ReLU(6, name="conv_dw_11_pwrelu"))


    hde.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="same", name="conv2d_2"))
    hde.add(layers.Flatten())
    hde.add(layers.Dense(2, activation="sigmoid", name="out"))
    hde.summary()

    return hde

if __name__ == '__main__':
    print ("DATA FORMAT: ", keras.backend.image_data_format())

    model = create_model()
    model.save("net.hdf5")

    np.random.seed(datetime.datetime.now().microsecond)

    for i in range(10):
        x = np.random.rand(1,192,192,3)
        print(x.shape)
        r = model.predict(x, batch_size=1)
        print(r)

    r = np.array([r])
    #x =  x.transpose(0, 3, 1, 2)
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
        if (len(data) > 0):
                
            # If this is a batch normalization layer, keep writing to last bin. Else, start new one
            if isinstance(layer, layers.BatchNormalization):
                data = data[1:]
            else:
                f.close()
                f = open("test_deer_hde/layers/" + layer.name + ".bin", mode='wb')
                if not layer.use_bias:
                    data.append(np.zeros(layer.output_shape[-1]))

            flattened_data = [d.flatten() for d in data]
            bin_write(f, np.concatenate(flattened_data))
    
    f.close()

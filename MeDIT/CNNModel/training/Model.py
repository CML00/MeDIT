from keras.layers import Conv2D, MaxPool2D, Input, Flatten, Dense, Dropout, PReLU, UpSampling2D
from keras.layers import BatchNormalization, Activation, Concatenate, Add, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

import numpy as np

def Conv2D_BN(x, filters, filter_size, padding='same', strides=(1, 1)):
    x = Conv2D(filters, filter_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = PReLU()(x)
    return x

def Conv2D_BN_Tanh(x, filters, filter_size, padding='same', strides=(1, 1)):
    x = Conv2D(filters, filter_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('tanh')(x)
    return x

def EncodingPart(input_shape, filters=64, blocks=3):
    inputs = Input(input_shape)
    x = inputs
    encoding_list = []
    for index in range(blocks):
        x_add = Conv2D_BN(x, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x_add, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        x = Add()([x, x_add])
        encoding_list.append(x)
        x = MaxPool2D()(x)
    return inputs, x, encoding_list

def DecodingPart(x, encoding_list, filters=64, blocks=3):
    for index in np.arange(blocks - 1, -1, -1):
        concat_list = []
        for temp in encoding_list:
            concat_list.append(temp[index])
        concat_list.append(x)

        concat = Concatenate(axis=-1)(concat_list)
        x_add = Conv2D_BN(concat, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x_add, filters * np.power(2, index), (3, 3))
        x = Conv2D_BN(x, filters * np.power(2, index), (1, 1))
        x = Add()([x, x_add])
        if index > 0:
            x = UpSampling2D()(x)
    return x

def ConnectionPart(encoding_output_list, filters=64, blocks=3):
    concat = Concatenate(axis=-1)(encoding_output_list)
    x_add = Conv2D_BN(concat, filters * np.power(2, blocks), (3, 3))
    x = Conv2D_BN(x_add, filters * np.power(2, blocks), (3, 3))
    x = Conv2D_BN(x, filters * np.power(2, blocks), (1, 1))
    x = Add()([x, x_add])
    x = UpSampling2D()(x)
    return x

def TrumpetNet(input_shape_list, filters=64, blocks=3):
    inputs_list, encoding_output_list, encoding_list = [], [], []
    for input_shape in input_shape_list:
        inputs, encoding_output, encoding = EncodingPart(input_shape, filters, blocks)
        inputs_list.append(inputs)
        encoding_output_list.append(encoding_output)
        encoding_list.append(encoding)

    connection = ConnectionPart(encoding_output_list, filters, blocks)

    x = DecodingPart(connection, encoding_list, filters, blocks)

    x = Conv2D_BN_Tanh(x, filters, (1, 1))
    x = Conv2D(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=inputs_list, outputs=x)


# model = TrumpetNet([(96, 96, 3) for index in range(3)], blocks=1)
# model.summary()
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file=r'C:\Users\SY\Desktop\model.png', show_shapes=True)


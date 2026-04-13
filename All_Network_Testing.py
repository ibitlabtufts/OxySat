# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 08:37:52 2025

@author: iBIT
"""

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from PIL import Image
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from tensorflow.keras import backend as K

from keras.models import *
from keras.layers import *
from keras.optimizers import *


##############################################################
'''
Useful blocks to build Unet

conv - BN - Activation - conv - BN - Activation - Dropout (if enabled)

'''


def conv_block(x, filter_size, size, dropout, batch_norm=False):

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv


def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).

    1. conv - BN - Activation - conv - BN - Activation
                                          - shortcut  - BN - shortcut+BN

    2. conv - BN - Activation - conv - BN
                                     - shortcut  - BN - shortcut+BN - Activation

    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    #conv = Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=3)(shortcut)

    res_path = add([shortcut, conv])
    res_path = Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    UNet,

    '''
    # network structure
    FILTER_NUM = 64 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters


    inputs = Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers

    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7

    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8

    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9

    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers

    conv_final = Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=3)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model
    model = Model(inputs, conv_final, name="UNet")
    return model

def Attention_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet,

    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    inputs = Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=3)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = Model(inputs, conv_final, name="Attention_UNet")
    
    return model

def ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Rsidual UNet, with attention

    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    inputs = Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, conv_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, conv_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, conv_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, conv_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers

    conv_final = Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = Model(inputs, conv_final, name="ResUNet")
    
    return model

def Attention_ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Rsidual UNet, with attention

    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    inputs = Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers

    conv_final = Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = Model(inputs, conv_final, name="AttentionResUNet")
    
    return model


from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Reshape, Permute
from keras.models import Model
from keras.layers import ConvLSTM2D
import tensorflow as tf
import os
import numpy as np
from scipy.io import loadmat
from tensorflow import keras
from tensorflow.keras import layers
import h5py
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, concatenate, Conv2DTranspose, Multiply

def recurrent_residual_block(input_tensor, num_filters):
    # Convolutional layers
    conv1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(conv1)
    
    # Reshape input tensor for ConvLSTM2D layer
    input_lstm = Reshape((1, input_tensor.shape[1], input_tensor.shape[2], num_filters))(conv2)
    input_lstm = Permute((1, 2, 3, 4))(input_lstm)
    
    # Recurrent layer
    convlstm = ConvLSTM2D(filters=num_filters, kernel_size=(3, 3), padding='same', return_sequences=True)(input_lstm)
    
    # Reshape output tensor for concatenation
    convlstm_reshaped = Reshape((input_tensor.shape[1], input_tensor.shape[2], num_filters))(convlstm)
    
    # Residual connection
    residual = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    residual_added = concatenate([convlstm_reshaped, residual], axis=-1)
    
    return residual_added

def unet_with_recurrent_residual_blocks(input_shape=(128, 256, 1), num_classes=1):
    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    
    # Encoder
    conv1 = recurrent_residual_block(inputs1, 64)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = recurrent_residual_block(pool1, 128)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = recurrent_residual_block(pool2, 256)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    conv4 = recurrent_residual_block(pool3, 512)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    # Encoder_US
    conv1u = recurrent_residual_block(inputs2, 64)
    pool1u = MaxPooling2D((2, 2))(conv1u)
    
    conv2u = recurrent_residual_block(pool1u, 128)
    pool2u = MaxPooling2D((2, 2))(conv2u)
    
    conv3u = recurrent_residual_block(pool2u, 256)
    pool3u = MaxPooling2D((2, 2))(conv3u)
    
    conv4u = recurrent_residual_block(pool3u, 512)
    pool4u = MaxPooling2D((2, 2))(conv4u)
    
    conv5 = recurrent_residual_block(pool4, 1024)
    
    # Concatenate
    convm = concatenate([conv5, pool4u])
    
    # Decoder
    up6 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=-1)
    conv6 = recurrent_residual_block(up6, 512)
    
    up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=-1)
    conv7 = recurrent_residual_block(up7, 256)
    
    up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=-1)
    conv8 = recurrent_residual_block(up8, 128)
    
    up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=-1)
    conv9 = recurrent_residual_block(up9, 64)
    
    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=[inputs1], outputs=[outputs])

    return Model([inputs1, inputs2], outputs)


def unet_with_recurrent_residual_blocks_wo_US(input_shape=(128, 256, 1), num_classes=1):
    inputs1 = Input(input_shape)
    
    # Encoder
    conv1 = recurrent_residual_block(inputs1, 64)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = recurrent_residual_block(pool1, 128)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = recurrent_residual_block(pool2, 256)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    conv4 = recurrent_residual_block(pool3, 512)
    pool4 = MaxPooling2D((2, 2))(conv4)
        
    conv5 = recurrent_residual_block(pool4, 1024)
        
    # Decoder
    up6 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=-1)
    conv6 = recurrent_residual_block(up6, 512)
    
    up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=-1)
    conv7 = recurrent_residual_block(up7, 256)
    
    up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=-1)
    conv8 = recurrent_residual_block(up8, 128)
    
    up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=-1)
    conv9 = recurrent_residual_block(up9, 64)
    
    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=[inputs1], outputs=[outputs])

    return Model(inputs1, outputs)



def DenseBlock(channels,inputs):

    conv1_1 = Conv2D(channels, (1, 1),activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1_1=BatchActivate(conv1_1)
    conv1_2 = Conv2D(channels//4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv1_1)
    conv1_2 = BatchActivate(conv1_2)

    conv2=concatenate([inputs,conv1_2])
    conv2_1 = Conv2D(channels, (1, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2_1 = BatchActivate(conv2_1)
    conv2_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv2_1)
    conv2_2 = BatchActivate(conv2_2)

    conv3 = concatenate([inputs, conv1_2,conv2_2])
    conv3_1 = Conv2D(channels, (1, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3_1 = BatchActivate(conv3_1)
    conv3_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv3_1)
    conv3_2 = BatchActivate(conv3_2)

    conv4 = concatenate([inputs, conv1_2, conv2_2,conv3_2])
    conv4_1 = Conv2D(channels, (1, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv4)
    conv4_1 = BatchActivate(conv4_1)
    conv4_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv4_1)
    conv4_2 = BatchActivate(conv4_2)
    result=concatenate([inputs,conv1_2, conv2_2,conv3_2,conv4_2])
    return result

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def DenseUNet(input_tensor1, start_neurons=32):

    #inputs = Input(input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(input_tensor1)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(start_neurons * 2, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(start_neurons * 4, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = DenseBlock(start_neurons * 8, pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)

    convm = DenseBlock(start_neurons * 16, pool4)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv4)
    uconv4 = BatchActivate(uconv4)
    uconv4 = DenseBlock(start_neurons * 8, uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(start_neurons * 4, uconv3)


    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = DenseBlock(start_neurons * 2, uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(start_neurons * 1, uconv1)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same",kernel_initializer='he_normal', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer

################### 2-input Dense U-Net
def CONFIGUN(input_tensor1, input_tensor2, start_neurons=32):

    #inputs = Input(input_size)
    #conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(input_tensor1)
    conv1 = res_conv_block(input_tensor1, 3, start_neurons * 1, 0.0, True)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    # Addition of US
    #us_l = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(input_tensor2)
    us_l = res_conv_block(input_tensor2, 3, start_neurons * 1, 0.0, True)
    us_l = BatchActivate(us_l)
    us_l = DenseBlock(start_neurons * 1, us_l)
    
    # Concatenate
    conv1 = concatenate([conv1, us_l])
    
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1u = MaxPooling2D((2, 2))(us_l)

    #conv2 = DenseBlock(start_neurons * 2, pool1)
    conv2 = res_conv_block(pool1, 3, start_neurons * 2, 0.0, True)
    #conv2u = DenseBlock(start_neurons * 2, pool1u)
    conv2u = res_conv_block(pool1u, 3, start_neurons * 2, 0.0, True)
    # Concatenate
    conv2 = concatenate([conv2, conv2u])
    
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2u = MaxPooling2D((2, 2))(conv2u)

    conv3 = DenseBlock(start_neurons * 4, pool2)
    conv3u = DenseBlock(start_neurons * 4, pool2u)
    # Concatenate
    conv3 = concatenate([conv3, conv3u])
    
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3u = MaxPooling2D((2, 2))(conv3u)

    conv4 = DenseBlock(start_neurons * 8, pool3)
    conv4u = DenseBlock(start_neurons * 8, pool3u)
    # Concatenate
    conv4 = concatenate([conv4, conv4u])
    
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4u = MaxPooling2D((2, 2))(conv4u)

    #convm = DenseBlock(start_neurons * 16, pool4)
    convm = res_conv_block(pool4, 3, start_neurons * 16, 0.0, True)
    #convmu = DenseBlock(start_neurons * 16, pool4u)
    convmu = res_conv_block(pool4u, 3, start_neurons * 16, 0.0, True)
    
    # Concatenate
    convm = concatenate([convm, convmu])
    
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(convm)
    gating_4 = gating_signal(convm, start_neurons * 8, True)
    att4 = attention_block(conv4, gating_4, start_neurons * 8)
    uconv4 = concatenate([deconv4, att4])
    uconv4 = Conv2D(start_neurons * 8, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv4)
    uconv4 = BatchActivate(uconv4)
    uconv4 = DenseBlock(start_neurons * 8, uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv4)
    gating_3 = gating_signal(uconv4, start_neurons * 4, True)
    att3 = attention_block(conv3, gating_3, start_neurons * 4)
    uconv3 = concatenate([deconv3, att3])
    uconv3 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(start_neurons * 4, uconv3)


    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv3)
    gating_2 = gating_signal(uconv3, start_neurons * 2, True)
    att2 = attention_block(conv2, gating_2, start_neurons * 2)
    uconv2 = concatenate([deconv2, att2])
    uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchActivate(uconv2)
    #uconv2 = DenseBlock(start_neurons * 2, uconv2)
    uconv2 = res_conv_block(uconv2, 3, start_neurons * 2, 0.0, True)


    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv2)
    gating_1 = gating_signal(uconv2, start_neurons * 1, True)
    att1 = attention_block(conv1, gating_1, start_neurons * 1)
    uconv1 = concatenate([deconv1, att1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchActivate(uconv1)
    #uconv1 = DenseBlock(start_neurons * 1, uconv1)
    uconv1 = res_conv_block(uconv1, 3, start_neurons * 1, 0.0, True)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same",kernel_initializer='he_normal', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer

def progressive_denoising_model(input_shape, stages=1):

    inputs1 = layers.Input(shape=input_shape)
    x1 = inputs1
    inputs2 = layers.Input(shape=input_shape)
    x2 = inputs2
    
    outputs = CONFIGUN(x1, x2)

    # Return the outputs from all stages
    return Model([inputs1, inputs2], outputs)

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from PIL import Image
import skimage.io as io
import skimage.transform as trans
from keras.callbacks import LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Reshape, Permute
from keras.models import Model
from keras.layers import ConvLSTM2D
from scipy.io import loadmat
from tensorflow import keras
from tensorflow.keras import layers
import h5py
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, concatenate, Conv2DTranspose
import pywt
##############################################################
'''
Useful blocks to build Unet

conv - BN - Activation - conv - BN - Activation - Dropout (if enabled)

'''
def conv_block(x, filter_size, size, dropout, batch_norm=False):

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv


def repeat_elem(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):

    conv = Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    #conv = Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=3)(shortcut)

    res_path = add([shortcut, conv])
    res_path = Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):

    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn


def DenseBlock(channels,inputs):

    conv1_1 = Conv2D(channels, (1, 1),activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1_1=BatchActivate(conv1_1)
    conv1_2 = Conv2D(channels//4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv1_1)
    conv1_2 = BatchActivate(conv1_2)

    conv2=concatenate([inputs,conv1_2])
    conv2_1 = Conv2D(channels, (1, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2_1 = BatchActivate(conv2_1)
    conv2_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv2_1)
    conv2_2 = BatchActivate(conv2_2)

    conv3 = concatenate([inputs, conv1_2,conv2_2])
    conv3_1 = Conv2D(channels, (1, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3_1 = BatchActivate(conv3_1)
    conv3_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv3_1)
    conv3_2 = BatchActivate(conv3_2)

    conv4 = concatenate([inputs, conv1_2, conv2_2,conv3_2])
    conv4_1 = Conv2D(channels, (1, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv4)
    conv4_1 = BatchActivate(conv4_1)
    conv4_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv4_1)
    conv4_2 = BatchActivate(conv4_2)
    result=concatenate([inputs,conv1_2, conv2_2,conv3_2,conv4_2])
    return result

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Define parameters
img_shape = (128, 256, 1)  # Adjust based on your image size and channels
batch_size = 4
epochs = 50
learning_rate = 1e-4

# Define paths
base_dir = "..\\Main_Test\\"
folders = ["LA6_1", "HA_1", "HA_2", "LA6_2", "LA6_1tr", "LA6_2tr"]
#folders = ["w1_main_low", "w2_main_low", "w1_supp_low", "w2_supp_low", "clean_1", "clean_2"]

# Function to load images from .mat file
def load_image(file_path):
    #data = loadmat(file_path)
    #image = data.get("I")  # Replace with actual key for image data in .mat file
    f = h5py.File(file_path)
    image = f['I']
    img = image[:]
    return img

# Prepare dataset lists
very_noisy_images = []
less_noisy_images = []
medium_noisy_images = []
clean_images = []

# Loop through each noise level folder and load images
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        images.append(np.array(load_image(file_path)))

    # Append to respective list based on folder
    if folder == "LA6_1":
        law1l = images
    elif folder == "LA6_2":
        law2l = images
    elif folder == "LA6_1tr":
        law2la = images
    elif folder == "LA6_2tr":
        law1la = images
    elif folder == "HA_1":
        clean_images1 = images
    elif folder == "HA_2":
        clean_images2 = images
    elif folder == "LA_w2":
        us = images

# Convert lists to numpy arrays for easy access
law1l = np.array(law1l)
law2l = np.array(law2l)
law1la = np.array(law1la)
law2la = np.array(law2la)
clean_images1 = np.array(clean_images1)
clean_images2 = np.array(clean_images2)

law1l=law1l[:,:,:,np.newaxis]
law2l=law2l[:,:,:,np.newaxis]
law1la=law1la[:,:,:,np.newaxis]
law2la=law2la[:,:,:,np.newaxis]
clean_images1=clean_images1[:,:,:,np.newaxis]
clean_images2=clean_images2[:,:,:,np.newaxis]

#################### Custom loss function ######################
import tensorflow as tf

# Intensity-aware weighted loss
def intensity_aware_loss(y_true, y_pred):
    # Calculate intensity at each pixel
    intensity = tf.abs(y_true)  # or some other function for intensity calculation
    weights = 1 / (1 + intensity)  # Higher weights for low-intensity regions
    
    # Calculate pixel-wise squared error
    loss = tf.square(y_true - y_pred)
    
    # Weight the loss
    weighted_loss = loss * weights
    
    total_loss = weighted_loss
    
    # Return mean loss
    return total_loss

def frequency_loss(y_true, y_pred):
    y_true_fft = tf.signal.fft2d(tf.cast(y_true, tf.complex64))
    y_pred_fft = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))
    loss = tf.reduce_mean(tf.abs(tf.abs(y_true_fft) - tf.abs(y_pred_fft)))
    return loss

def gradient_loss(y_true, y_pred):
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))

# Weighted Loss with Gradient Loss
def total_loss(y_true, y_pred, lambda_weight=0.9, lambda_freq=0.1, ssim_wt=0.05, snr_wt=0.05, le_wt=0.3):
    weight_loss = intensity_aware_loss(y_true, y_pred)
    freq_loss = frequency_loss(y_true, y_pred)
    grad_loss = gradient_loss(y_true, y_pred)
    
    # Compute Signal-to-Noise Ratio (SNR)
    snr = tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=1.0))
    
    # Compute SSIM
    ssim = tf.reduce_mean(tf.image.ssim(y_pred, y_true, max_val=1.0))
    
    # Compute error between true and predicted images
    error = y_true - y_pred

    # Compute Lyapunov Exponent (LE) approximation using local sensitivity
    le = tf.reduce_mean(tf.abs(error[:, 1:, :, :] - error[:, :-1, :, :]))  # Temporal/Spatial sensitivity
    
    return lambda_weight * weight_loss + lambda_freq * freq_loss
    
      

##################################### Test
img_shape = (128, 256, 1)  # Adjust based on your image size and channels

# Our model
model = progressive_denoising_model(img_shape)

model = load_model('CONFIGUN_W1.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict([law1l, law1la])
np.save("Our_Model_W1", den_data)

model = load_model('CONFIGUN_W2.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict([law2l, law2la])
np.save("Our_Model_W2", den_data)


# Dense_MSE
model = progressive_denoising_model(img_shape)

model = load_model('DenseUN_model_with_US_6_epch_30_lr_5e3_MSE_loss_wave1_intermingled_att_res.hdf5', custom_objects={'intensity_aware_loss': intensity_aware_loss})
den_data = model.predict(law1l)
np.save("Dense_MSE_W1", den_data)

model = load_model('DenseUN_model_with_US_6_epch_30_lr_5e3_MSE_loss_wave2_intermingled_att_res.hdf5', custom_objects={'intensity_aware_loss': intensity_aware_loss})
den_data = model.predict(law2l)
np.save("Dense_MSE_W2", den_data)


# Dense_Only
model = load_model('Dense_UNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Dense_Only_W1", den_data)

model = load_model('Dense_UNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Dense_Only_W2", den_data)



model = load_model('Attention_ResUNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Attention_ResUNet_W1", den_data)

model = load_model('Attention_ResUNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Attention_ResUNet_W2", den_data)



model = load_model('Attention_UNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Attention_UNet_W1", den_data)

model = load_model('Attention_UNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Attention_UNet_W2", den_data)



model = load_model('ResUNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("ResUNet_W1", den_data)

model = load_model('ResUNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("ResUNet_W2", den_data)



model = load_model('UNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("UNet_W1", den_data)

model = load_model('UNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("UNet_W2", den_data)

########## cGAN
d = tf.image.resize(law1l, size=(256, 256), method='bilinear')  # (700, 256, 256, 1)
d = (d * 2) - 1
model = load_model('cGAN_model.hdf5')
den_data = model.predict(d)
den_data = (den_data - den_data.min())/(den_data.max()-den_data.min())
np.save("DenP2P_W1", den_data)


d = tf.image.resize(law2l, size=(256, 256), method='bilinear')  # (700, 256, 256, 1)
d = (d * 2) - 1
model = load_model('cGAN_model.hdf5')
den_data = model.predict(d)
den_data = (den_data - den_data.min())/(den_data.max()-den_data.min())
np.save("DenP2P_W2", den_data)


np.save("clean_W1", clean_images1)
np.save("clean_W2", clean_images2)

np.save("noisy_W1", law1l)
np.save("noisy_W2", law2l)



################# Tube data

# Define paths
base_dir = "..\\Test_Tube\\"
folders = ["W1", "W2", "H1", "H2"]

# Function to load images from .mat file
def load_image(file_path):
    #data = loadmat(file_path)
    #image = data.get("I")  # Replace with actual key for image data in .mat file
    f = h5py.File(file_path)
    image = f['I']
    img = image[:]
    return img

# Prepare dataset lists
very_noisy_images = []
less_noisy_images = []
medium_noisy_images = []
clean_images = []

# Loop through each noise level folder and load images
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        images.append(np.array(load_image(file_path)))

    # Append to respective list based on folder
    if folder == "W1":
        law1l = images
    elif folder == "W2":
        law2l = images
    elif folder == "H1":
        clean_images1 = images
    elif folder == "H2":
        clean_images2 = images


# Convert lists to numpy arrays for easy access
law1l = np.array(law1l)
law2l = np.array(law2l)
clean_images1 = np.array(clean_images1)
clean_images2 = np.array(clean_images2)

law1l=law1l[:,:,:,np.newaxis]
law2l=law2l[:,:,:,np.newaxis]
clean_images1=clean_images1[:,:,:,np.newaxis]
clean_images2=clean_images2[:,:,:,np.newaxis]


model = load_model('CONFIGUN_W1.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict([law1l, law2l])
den_data[den_data<0.1] = 0
np.save("Tube_Our_Model_W1", den_data)

model = load_model('CONFIGUN_W2.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict([law2l, law1l])
den_data[den_data<0.1] = 0
np.save("Tube_Our_Model_W2", den_data)


# Dense_MSE
model = progressive_denoising_model(img_shape)

model = load_model('DenseUN_model_with_US_6_epch_30_lr_5e3_MSE_loss_wave1_intermingled_att_res.hdf5', custom_objects={'intensity_aware_loss': intensity_aware_loss})
den_data = model.predict(law1l)
np.save("Tube_Dense_MSE_W1", den_data)

model = load_model('DenseUN_model_with_US_6_epch_30_lr_5e3_MSE_loss_wave2_intermingled_att_res.hdf5', custom_objects={'intensity_aware_loss': intensity_aware_loss})
den_data = model.predict(law2l)
np.save("Tube_Dense_MSE_W2", den_data)


# Dense_Only
model = load_model('Dense_UNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Tube_Dense_Only_W1", den_data)

model = load_model('Dense_UNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Tube_Dense_Only_W2", den_data)



model = load_model('Attention_ResUNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Tube_Attention_ResUNet_W1", den_data)

model = load_model('Attention_ResUNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Tube_Attention_ResUNet_W2", den_data)



model = load_model('Attention_UNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Tube_Attention_UNet_W1", den_data)

model = load_model('Attention_UNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Tube_Attention_UNet_W2", den_data)



model = load_model('ResUNet_wo_US_w1_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Tube_ResUNet_W1", den_data)

model = load_model('ResUNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Tube_ResUNet_W2", den_data)



model = load_model('UNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law1l)
np.save("Tube_UNet_W1", den_data)

model = load_model('UNet_wo_US_w2_total_loss.hdf5', custom_objects={'total_loss': total_loss})
den_data = model.predict(law2l)
np.save("Tube_UNet_W2", den_data)


d = tf.image.resize(law1l, size=(256, 256), method='bilinear')  # (700, 256, 256, 1)
d = (d * 2) - 1
model = load_model('cGAN_model.hdf5')
den_data = model.predict(d)

den_data = (den_data - den_data.min())/(den_data.max()-den_data.min())
den_data[0:3,0:10, 0:256] = 0.00200993
np.save("Tube_cGAN_W1", den_data)

plt.imshow(np.squeeze(den_data[0]), cmap='hot')


d = tf.image.resize(law2l, size=(256, 256), method='bilinear')  # (700, 256, 256, 1)
d = (d * 2) - 1
model = load_model('cGAN_model.hdf5')
den_data = model.predict(d)

den_data = (den_data - den_data.min())/(den_data.max()-den_data.min())
den_data[0:3,0:10, 0:256] = 0.00200993
np.save("Tube_cGAN_W2", den_data)

plt.imshow(np.squeeze(den_data[0]), cmap='hot')

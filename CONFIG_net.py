# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:47:36 2025

@author: iBIT
"""

####################################################### Our developed network ###########################################
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

##############################################################
'''
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

################### 2-input Dense U-Net
def CONFIGUN(input_tensor1, input_tensor2, start_neurons=32):

    #inputs = Input(input_size)
    conv1 = res_conv_block(input_tensor1, 3, start_neurons * 1, 0.0, True)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    # Addition of Input 2
    us_l = res_conv_block(input_tensor2, 3, start_neurons * 1, 0.0, True)
    us_l = BatchActivate(us_l)
    us_l = DenseBlock(start_neurons * 1, us_l)
    
    # Concatenate
    conv1 = concatenate([conv1, us_l])
    
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1u = MaxPooling2D((2, 2))(us_l)

    conv2 = res_conv_block(pool1, 3, start_neurons * 2, 0.0, True)
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

    convm = res_conv_block(pool4, 3, start_neurons * 16, 0.0, True)
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
    uconv2 = res_conv_block(uconv2, 3, start_neurons * 2, 0.0, True)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv2)
    gating_1 = gating_signal(uconv2, start_neurons * 1, True)
    att1 = attention_block(conv1, gating_1, start_neurons * 1)
    uconv1 = concatenate([deconv1, att1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = res_conv_block(uconv1, 3, start_neurons * 1, 0.0, True)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same",kernel_initializer='he_normal', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer

######################### 1 layers
def DenseUNet_1L(input_tensor1, input_tensor2, start_neurons=32):

    #inputs = Input(input_size)
    conv1 = res_conv_block(input_tensor1, 3, start_neurons * 1, 0.0, True)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    # Addition of Input 2
    us_l = res_conv_block(input_tensor2, 3, start_neurons * 1, 0.0, True)
    us_l = BatchActivate(us_l)
    us_l = DenseBlock(start_neurons * 1, us_l)
    
    # Concatenate
    conv1 = concatenate([conv1, us_l])
    
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1u = MaxPooling2D((2, 2))(us_l)
    
    convm = res_conv_block(pool1, 3, start_neurons * 4, 0.0, True)
    convmu = res_conv_block(pool1u, 3, start_neurons * 4, 0.0, True)
    
    # Concatenate
    convm = concatenate([convm, convmu])

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(convm)
    gating_1 = gating_signal(convm, start_neurons * 1, True)
    att4 = attention_block(conv1, gating_1, start_neurons * 8)
    uconv1 = concatenate([deconv1, att4])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(start_neurons * 1, uconv1)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same",kernel_initializer='he_normal', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer

def DenseUNet_2L(input_tensor1, input_tensor2, start_neurons=32):

    #inputs = Input(input_size)
    conv1 = res_conv_block(input_tensor1, 3, start_neurons * 1, 0.0, True)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    # Addition of Input 2
    us_l = res_conv_block(input_tensor2, 3, start_neurons * 1, 0.0, True)
    us_l = BatchActivate(us_l)
    us_l = DenseBlock(start_neurons * 1, us_l)
    
    # Concatenate
    conv1 = concatenate([conv1, us_l])
    
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1u = MaxPooling2D((2, 2))(us_l)

    conv2 = res_conv_block(pool1, 3, start_neurons * 2, 0.0, True)
    conv2u = res_conv_block(pool1u, 3, start_neurons * 2, 0.0, True)
    # Concatenate
    conv2 = concatenate([conv2, conv2u])
    
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2u = MaxPooling2D((2, 2))(conv2u)

    convm = res_conv_block(pool2, 3, start_neurons * 16, 0.0, True)
    convmu = res_conv_block(pool2u, 3, start_neurons * 16, 0.0, True)
    
    # Concatenate
    convm = concatenate([convm, convmu])
    
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(convm)
    gating_2 = gating_signal(convm, start_neurons * 2, True)
    att2 = attention_block(conv2, gating_2, start_neurons * 2)
    uconv2 = concatenate([deconv2, att2])
    uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = res_conv_block(uconv2, 3, start_neurons * 2, 0.0, True)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv2)
    gating_1 = gating_signal(uconv2, start_neurons * 1, True)
    att1 = attention_block(conv1, gating_1, start_neurons * 1)
    uconv1 = concatenate([deconv1, att1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = res_conv_block(uconv1, 3, start_neurons * 1, 0.0, True)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same",kernel_initializer='he_normal', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer

def DenseUNet_3L(input_tensor1, input_tensor2, start_neurons=32):

    #inputs = Input(input_size)
    conv1 = res_conv_block(input_tensor1, 3, start_neurons * 1, 0.0, True)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    # Addition of Input 2
    us_l = res_conv_block(input_tensor2, 3, start_neurons * 1, 0.0, True)
    us_l = BatchActivate(us_l)
    us_l = DenseBlock(start_neurons * 1, us_l)
    
    # Concatenate
    conv1 = concatenate([conv1, us_l])
    
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1u = MaxPooling2D((2, 2))(us_l)

    conv2 = res_conv_block(pool1, 3, start_neurons * 2, 0.0, True)
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

    convm = res_conv_block(pool3, 3, start_neurons * 16, 0.0, True)
    convmu = res_conv_block(pool3u, 3, start_neurons * 16, 0.0, True)
    
    # Concatenate
    convm = concatenate([convm, convmu])
    
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(convm)
    gating_3 = gating_signal(convm, start_neurons * 4, True)
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
    uconv2 = res_conv_block(uconv2, 3, start_neurons * 2, 0.0, True)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv2)
    gating_1 = gating_signal(uconv2, start_neurons * 1, True)
    att1 = attention_block(conv1, gating_1, start_neurons * 1)
    uconv1 = concatenate([deconv1, att1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = res_conv_block(uconv1, 3, start_neurons * 1, 0.0, True)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same",kernel_initializer='he_normal', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer

def progressive_denoising_model(input_shape, stages=1):

    inputs1 = layers.Input(shape=input_shape)
    x1 = inputs1
    inputs2 = layers.Input(shape=input_shape)
    x2 = inputs2
    
    #outputs = DenseUNet_1L(x1, x2)
    #outputs = DenseUNet_2L(x1, x2)
    outputs = CONFIGUN(x1, x2)

    # Return the outputs from all stages
    return Model([inputs1, inputs2], outputs)


# Define parameters
img_shape = (128, 256, 1)  # Adjust based on your image size and channels
batch_size = 4
epochs = 40
learning_rate = 1e-4

# Define paths
base_dir = "..\\Main_Train\\"
folders = ["w1_main_low", "w2_main_low", "w1_supp_low", "w2_supp_low", "clean_1", "clean_2"]

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
    if folder == "w1_main_low":
        law1l = images
    elif folder == "w2_main_low":
        law2l = images
    elif folder == "w1_supp_low":
        law2la = images
    elif folder == "w2_supp_low":
        law1la = images
    elif folder == "clean_1":
        clean_images1 = images
    elif folder == "clean_2":
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

# Create tf.data.Dataset for training
def load_progressive_dataset(batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((la96, clean_images))
    dataset = dataset.shuffle(buffer_size=len(la96)).batch(batch_size)
    return dataset

#################### Custom loss function ######################
import tensorflow as tf

# Intensity-aware weighted loss
def intensity_aware_loss(y_true, y_pred, mse_portion=0.5):
    # Calculate intensity at each pixel
    intensity = tf.abs(y_true)  # or some other function for intensity calculation
    weights = 1 / (1 + intensity)  # Higher weights for low-intensity regions
    
    # Calculate pixel-wise squared error
    loss = tf.square(y_true - y_pred)
    
    # Weight the loss
    weighted_loss = (1-mse_portion) * loss * weights + mse_portion*loss
    
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
def total_loss(y_true, y_pred, lambda_weight=0.99, lambda_freq=0.01, ssim_wt=0.05, snr_wt=0.05, le_wt=0.3):
    weight_loss = intensity_aware_loss(y_true, y_pred, mse_portion=0.5)
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
    
########################################################################################


#^^^^^^^^^^^^^^^^^^^^ Different metrics ^^^^^^^^^^^^^^^^^^^^
import tensorflow as tf

def lyapunov_exponent(y_true, y_pred):
    """
    Compute Lyapunov Exponent (LE) approximation using local sensitivity.
    Measures how small perturbations in pixel values evolve over spatial dimensions.
    """
    error = tf.abs(y_true - y_pred)
    le_x = tf.reduce_mean(tf.abs(error[:, 1:, :, :] - error[:, :-1, :, :]))  # Sensitivity along x-axis
    le_y = tf.reduce_mean(tf.abs(error[:, :, 1:, :] - error[:, :, :-1, :]))  # Sensitivity along y-axis
    return (le_x + le_y) / 2  # Average LE

def cassim(y_true, y_pred):
    """
    Chaos-Aware Structural Similarity Metric (CASSIM)
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    le = lyapunov_exponent(y_true, y_pred)
    return tf.exp(-le) * ssim  # Chaos-aware SSIM


def local_entropy(image, window_size=3):
    """
    Computes local Shannon entropy using a moving window.
    """
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, window_size, window_size, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )
    mean_patch = tf.reduce_mean(patches, axis=-1, keepdims=True)
    variance_patch = tf.reduce_mean(tf.square(patches - mean_patch), axis=-1, keepdims=True)
    entropy = -variance_patch * tf.math.log(variance_patch + 1e-8)  # Shannon entropy
    return tf.reduce_mean(entropy)

def box_counting_fractal_dim(image, threshold=0.5):
    """
    Computes an approximation of the fractal dimension using a box-counting method.
    """
    binary_image = tf.cast(image > threshold, tf.float32)
    box_sizes = [1, 2, 4, 8, 16]
    box_counts = []

    for size in box_sizes:
        pooled = tf.nn.avg_pool(binary_image, ksize=size, strides=size, padding='SAME')
        box_count = tf.reduce_sum(tf.cast(pooled > 0, tf.float32))
        box_counts.append(tf.math.log(box_count + 1e-8))

    log_scales = tf.math.log(tf.constant(box_sizes, dtype=tf.float32))
    D_f = -tf.reduce_mean(box_counts) / tf.reduce_mean(log_scales)  # Fractal dimension
    return D_f

def efd(y_true, y_pred):
    """
    Entropic Fractal Dimension (EFD) metric.
    """
    entropy = local_entropy(y_pred)
    fractal_dim = box_counting_fractal_dim(y_pred)
    return entropy / fractal_dim  # Complexity measure

def compute_laplacian(image):
    """
    Computes the discrete Laplacian operator for an image.
    """
    laplacian_filter = tf.constant(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32
    )
    laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])
    return tf.nn.conv2d(image, laplacian_filter, strides=1, padding='SAME')

def quantum_potential(image):
    """
    Computes an adaptive quantum potential V(x, y) based on edge strength.
    """
    grad_x = tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :])
    grad_y = tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :])

    grad_x = tf.pad(grad_x, [[0, 0], [1, 0], [0, 0], [0, 0]])  # Pad height
    grad_y = tf.pad(grad_y, [[0, 0], [0, 0], [1, 0], [0, 0]])  # Pad width 

    return tf.sqrt(tf.square(grad_x) + tf.square(grad_y))

def schrodinger_energy(image):
    """
    Computes the Schrödinger equation-based energy functional.
    """
    laplacian = compute_laplacian(image)
    potential = quantum_potential(image)
    return tf.reduce_mean(-0.5 * laplacian + potential * image)

def sif(y_true, y_pred):
    """
    Schrödinger Image Fidelity (SIF) metric.
    """
    E_ref = schrodinger_energy(y_true)
    E_dist = schrodinger_energy(y_pred)
    return 1 / (1 + tf.abs(E_ref - E_dist))  # Higher value means better quality

def compute_energy_spectrum(image):
    """
    Computes the Fourier-based energy spectrum of an image.
    """
    fft_image = tf.signal.fft2d(tf.cast(image, tf.complex64))
    power_spectrum = tf.abs(fft_image) ** 2
    return tf.reduce_mean(power_spectrum)

def twiq(y_true, y_pred):
    """
    Turbulence-Weighted Image Quality (TWIQ) metric.
    """
    energy_ref = compute_energy_spectrum(y_true)
    energy_dist = compute_energy_spectrum(y_pred)
    return tf.abs(energy_ref - energy_dist)  # Lower means better quality

################################################################

################ PSNR and SSIM over epoch
def cal_ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def cal_psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

################################# CONFIG W1 ###########################################
############################################################
####################### Intensity_aware_loss + Frequency loss
############################################################
model = progressive_denoising_model(img_shape)
noise_levels = [law1l[1:899]]  # List of noise-level datasets
seg_levels = [law1la[1:899]]  # List of noise-level datasets
targets = [clean_images1[1:899]]  # Corresponding targets

# For validation
noise_Vlevels = [law1l[900:998]]  # List of noise-level datasets
seg_Vlevels = [law1la[900:998]]  # List of noise-level datasets
targetsV = [clean_images1[900:998]]  # Corresponding targets
    
# Compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(5e-3), loss=total_loss, metrics=[cassim, efd, sif, twiq])

# Create a generator to feed noise-level pairs into the model
def data_generator(noise_levels, seg_lvl, targets, batch_size=2):
    while True:
        idx = np.random.randint(0, len(noise_levels[0]), batch_size)
        inputs1 = noise_levels[0][idx]
        inputs2 = seg_lvl[0][idx]
        outputs = [targets[i][idx] for i in range(len(targets))]
        yield [inputs1, inputs2], outputs

# Train the model
batch_size = 4
steps_per_epoch = len(law1l) // batch_size
history = model.fit(
    data_generator(noise_levels, seg_levels, targets, batch_size),
    steps_per_epoch=steps_per_epoch,
    validation_split=0.1,
    validation_data=data_generator(noise_Vlevels, seg_Vlevels, targetsV, batch_size), validation_batch_size=2,
    validation_steps=steps_per_epoch,
    validation_freq=1,
    epochs=40,
)

model.save("CONFIGUN_W1.hdf5")

import pandas as pd
history_df = pd.DataFrame(history.history)

# Save history to CSV file
history_df.to_csv('CONFIGUN_W1.csv', index=False)

################################# CONFIG W2 ###########################################
############################################################
####################### Intensity_aware_loss + Frequency loss
############################################################
model2 = progressive_denoising_model(img_shape)
noise_levels = [law2l[1:899]]  # List of noise-level datasets
seg_levels = [law2la[1:899]]  # List of noise-level datasets
targets = [clean_images2[1:899]]  # Corresponding targets

# For validation
noise_Vlevels = [law2l[900:945]]  # List of noise-level datasets
seg_Vlevels = [law2la[900:945]]  # List of noise-level datasets
targetsV = [clean_images2[900:945]]  # Corresponding targets

# Compile the model
model2.compile(optimizer= tf.keras.optimizers.Adam(5e-3), loss=total_loss, metrics=[cassim, efd, sif, twiq])

# Create a generator to feed noise-level pairs into the model
def data_generator(noise_levels, seg_lvl, targets, batch_size=2):
    while True:
        idx = np.random.randint(0, len(noise_levels[0]), batch_size)
        inputs1 = noise_levels[0][idx]
        inputs2 = seg_lvl[0][idx]
        outputs = [targets[i][idx] for i in range(len(targets))]
        yield [inputs1, inputs2], outputs

# Train the model
batch_size = 4
steps_per_epoch = len(law2l) // batch_size
history = model2.fit(
    data_generator(noise_levels, seg_levels, targets, batch_size),
    steps_per_epoch=steps_per_epoch,
    validation_split=0.1,
    validation_data=data_generator(noise_Vlevels, seg_Vlevels, targetsV, batch_size), validation_batch_size=2,
    validation_steps=steps_per_epoch,
    validation_freq=1,
    epochs=40,
)

model2.save("CONFIGUN_W2.hdf5")

import pandas as pd
history_df2 = pd.DataFrame(history.history)

# Save history to CSV file
history_df2.to_csv('CONFIGUN_W2.csv', index=False)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulating example data for history_df.loss and history_df.val_loss
# (since no actual data was provided)
epochs = np.arange(1, 40)
loss = history_df.loss
val_loss = history_df.val_loss

# Plotting
plt.figure(figsize=(10, 6), facecolor='white')
plt.plot(loss, label='Training Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='s')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

loss = history_df2.loss
val_loss = history_df2.val_loss

# Plotting
plt.figure(figsize=(10, 6), facecolor='white')
plt.plot(loss, label='Training Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='s')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##################################

'''
################### Tube testing
base_dir = "..\\Test_Tube\\"
folders = ["W1", "W2", "H1", "H2"]

# Function to load images from .mat file
def load_image(file_path):
    f = h5py.File(file_path)
    image = f['I']
    img = image[:]
    return img

# Loop through each noise level folder and load images
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        images.append(np.array(load_image(file_path)))

    if folder == "W1":
        law1l = images
    elif folder == "W2":
        law2l = images
    elif folder == "H1":
        clean_images1 = images
    elif folder == "H2":
        clean_images2 = images

law1l = np.array(law1l)
law2l = np.array(law2l)

law1l=law1l[:,:,:,np.newaxis]
law2l=law2l[:,:,:,np.newaxis]

den_data = model.predict([law1l, law2l])
den_data[den_data<0.15] = 0
np.save("Tube_Our_Model_W1", den_data)

den_data2 = model2.predict([law2l, law1l])
den_data2[den_data2<0.1] = 0
np.save("Tube_Our_Model_W2", den_data2)
'''

'''
denoised_image = model.predict([noise_levels, seg_levels])
tr=tf.cast(targets, dtype='float64')
de=tf.cast(denoised_image, dtype='float64')
    
# Calculate complexity metrics
ssim_s = np.mean(cal_ssim(tr[0], de[0]))
psnr_s = np.mean(cal_psnr(tr[0], de[0]))

np.save("SSIM_CONFIG_wave1",ssim_all)
np.save("PSNR_CONFIG_wave1",psnr_all)
'''

'''
denoised_image = model.predict([noise_levels, seg_levels])
tr=tf.cast(targets, dtype='float64')
de=tf.cast(denoised_image, dtype='float64')
    
# Calculate complexity metrics
ssim_s = np.mean(cal_ssim(tr[0], de[0]))
psnr_s = np.mean(cal_psnr(tr[0], de[0]))

np.save("SSIM_CONFIG_wave2",ssim_all)
np.save("PSNR_CONFIG_wave2",psnr_all)
'''
#########################################################################################################################

'''
########################################### Same architecture with different loss #######################################
############################################################
####################### Intensity_aware_loss
############################################################
model = progressive_denoising_model(img_shape)
noise_levels = [law1l]  # List of noise-level datasets
seg_levels = [law1la]  # List of noise-level datasets
targets = [clean_images1]  # Corresponding targets
    
# Compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(5e-3), loss=intensity_aware_loss, metrics=[cassim, efd, sif, twiq])

# Create a generator to feed noise-level pairs into the model
def data_generator(noise_levels, seg_lvl, targets, batch_size=2):
    while True:
        idx = np.random.randint(0, len(noise_levels[0]), batch_size)
        inputs1 = noise_levels[0][idx]
        inputs2 = seg_lvl[0][idx]
        outputs = [targets[i][idx] for i in range(len(targets))]
        yield [inputs1, inputs2], outputs

# Define checkpoint directory
checkpoint_path = "C:\\Users\\iBIT\\Avijit\\OxySat_Train\\Low_Intensity_restore\\MAT\\W2\\checkpt\\w1_denoising_model_epoch_{epoch:02d}.h5"

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # Saves only weights (set False to save full model)
    save_best_only=False,    # Set True if you want only the best model
    save_freq='epoch',       # Save after each epoch
    verbose=1
)

# Train the model
batch_size = 4
steps_per_epoch = len(law1l) // batch_size
history = model.fit(
    data_generator(noise_levels, seg_levels, targets, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=40,
    #callbacks=[checkpoint_callback]
)

model.save("CONFIG_wave1_hlfMSE_hlfWeightAdj.hdf5")

import pandas as pd
history_df = pd.DataFrame(history.history)

# Save history to CSV file
history_df.to_csv('CONFIG_wave1_hlfMSE_hlfWeightAdj.csv', index=False)

denoised_image = model.predict([noise_levels, seg_levels])
tr=tf.cast(targets, dtype='float64')
de=tf.cast(denoised_image, dtype='float64')
    
# Calculate complexity metrics
ssim_s = np.mean(cal_ssim(tr[0], de[0]))
psnr_s = np.mean(cal_psnr(tr[0], de[0]))

np.save("SSIM_CONFIG_wave1_hlfMSE_hlfWeightAdj",ssim_all)
np.save("PSNR_CONFIG_wave1_hlfMSE_hlfWeightAdj",psnr_all)

################################# CONFIG W2 ###########################################
model = progressive_denoising_model(img_shape)
noise_levels = [law2l]  # List of noise-level datasets
seg_levels = [law2la]  # List of noise-level datasets
targets = [clean_images2]  # Corresponding targets
    
# Compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(5e-3), loss=intensity_aware_loss, metrics=[cassim, efd, sif, twiq])


# Create a generator to feed noise-level pairs into the model
def data_generator(noise_levels, seg_lvl, targets, batch_size=2):
    while True:
        idx = np.random.randint(0, len(noise_levels[0]), batch_size)
        inputs1 = noise_levels[0][idx]
        inputs2 = seg_lvl[0][idx]
        outputs = [targets[i][idx] for i in range(len(targets))]
        yield [inputs1, inputs2], outputs

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Define checkpoint directory
checkpoint_path = "C:\\Users\\iBIT\\Avijit\\OxySat_Train\\Low_Intensity_restore\\MAT\\W2\\checkpt\\w2_denoising_model_epoch_{epoch:02d}.h5"

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # Saves only weights (set False to save full model)
    save_best_only=False,    # Set True if you want only the best model
    save_freq='epoch',       # Save after each epoch
    verbose=1
)

# Train the model
batch_size = 4
steps_per_epoch = len(law2l) // batch_size
history = model.fit(
    data_generator(noise_levels, seg_levels, targets, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=40,
    #callbacks=[checkpoint_callback]
)

model.save("CONFIG_wave2_hlfMSE_hlfWeightAdj.hdf5")

import pandas as pd
history_df = pd.DataFrame(history.history)

# Save history to CSV file
history_df.to_csv('CONFIG_wave2_hlfMSE_hlfWeightAdj.csv', index=False)

denoised_image = model.predict([noise_levels, seg_levels])
tr=tf.cast(targets, dtype='float64')
de=tf.cast(denoised_image, dtype='float64')
    
# Calculate complexity metrics
ssim_s = np.mean(cal_ssim(tr[0], de[0]))
psnr_s = np.mean(cal_psnr(tr[0], de[0]))

np.save("SSIM_CONFIG_wave2_hlfMSE_hlfWeightAdj",ssim_all)
np.save("PSNR_CONFIG_wave2_hlfMSE_hlfWeightAdj",psnr_all)

############################################################
####################### MSE
############################################################
model = progressive_denoising_model(img_shape)
noise_levels = [law1l]  # List of noise-level datasets
seg_levels = [law1la]  # List of noise-level datasets
targets = [clean_images1]  # Corresponding targets
    
# Compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(5e-3), loss='mse', metrics=[cassim, efd, sif, twiq])

# Create a generator to feed noise-level pairs into the model
def data_generator(noise_levels, seg_lvl, targets, batch_size=2):
    while True:
        idx = np.random.randint(0, len(noise_levels[0]), batch_size)
        inputs1 = noise_levels[0][idx]
        inputs2 = seg_lvl[0][idx]
        outputs = [targets[i][idx] for i in range(len(targets))]
        yield [inputs1, inputs2], outputs

# Define checkpoint directory
checkpoint_path = "C:\\Users\\iBIT\\Avijit\\OxySat_Train\\Low_Intensity_restore\\MAT\\W2\\checkpt\\w1_denoising_model_epoch_{epoch:02d}.h5"

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # Saves only weights (set False to save full model)
    save_best_only=False,    # Set True if you want only the best model
    save_freq='epoch',       # Save after each epoch
    verbose=1
)

# Train the model
batch_size = 4
steps_per_epoch = len(law1l) // batch_size
history = model.fit(
    data_generator(noise_levels, seg_levels, targets, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=40,
    #callbacks=[checkpoint_callback]
)

model.save("CONFIG_wave1_mse.hdf5")

import pandas as pd
history_df = pd.DataFrame(history.history)

# Save history to CSV file
history_df.to_csv('CONFIG_wave1_mse.csv', index=False)

denoised_image = model.predict([noise_levels, seg_levels])
tr=tf.cast(targets, dtype='float64')
de=tf.cast(denoised_image, dtype='float64')
    
# Calculate complexity metrics
ssim_s = np.mean(cal_ssim(tr[0], de[0]))
psnr_s = np.mean(cal_psnr(tr[0], de[0]))

np.save("SSIM_CONFIG_wave1_mse",ssim_all)
np.save("PSNR_CONFIG_wave1_mse",psnr_all)

################################# CONFIG W2 ###########################################
model = progressive_denoising_model(img_shape)
noise_levels = [law2l]  # List of noise-level datasets
seg_levels = [law2la]  # List of noise-level datasets
targets = [clean_images2]  # Corresponding targets
    
# Compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(5e-3), loss='mse', metrics=[cassim, efd, sif, twiq])


# Create a generator to feed noise-level pairs into the model
def data_generator(noise_levels, seg_lvl, targets, batch_size=2):
    while True:
        idx = np.random.randint(0, len(noise_levels[0]), batch_size)
        inputs1 = noise_levels[0][idx]
        inputs2 = seg_lvl[0][idx]
        outputs = [targets[i][idx] for i in range(len(targets))]
        yield [inputs1, inputs2], outputs

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Define checkpoint directory
checkpoint_path = "C:\\Users\\iBIT\\Avijit\\OxySat_Train\\Low_Intensity_restore\\MAT\\W2\\checkpt\\w2_denoising_model_epoch_{epoch:02d}.h5"

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # Saves only weights (set False to save full model)
    save_best_only=False,    # Set True if you want only the best model
    save_freq='epoch',       # Save after each epoch
    verbose=1
)

# Train the model
batch_size = 4
steps_per_epoch = len(law2l) // batch_size
history = model.fit(
    data_generator(noise_levels, seg_levels, targets, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=40,
    #callbacks=[checkpoint_callback]
)

model.save("CONFIG_wave2_mse.hdf5")

import pandas as pd
history_df = pd.DataFrame(history.history)

# Save history to CSV file
history_df.to_csv('CONFIG_wave2_mse.csv', index=False)

denoised_image = model.predict([noise_levels, seg_levels])
tr=tf.cast(targets, dtype='float64')
de=tf.cast(denoised_image, dtype='float64')
    
# Calculate complexity metrics
ssim_s = np.mean(cal_ssim(tr[0], de[0]))
psnr_s = np.mean(cal_psnr(tr[0], de[0]))

np.save("SSIM_CONFIG_wave2_mse",ssim_all)
np.save("PSNR_CONFIG_wave2_mse",psnr_all)

#########################################################################################################################
'''

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:50:59 2026

@author: iBIT
"""

import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from tensorflow.keras import backend as K
import h5py, os
from scipy.io import savemat
import matplotlib.pyplot as plt

#################### Custom loss function ######################
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
############ Data
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
################################################################

def calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy):

    # Calculate the difference in extinction coefficients
    Delta_ELambda1 = ELambda1_Doxy - ELambda1_oxy
    Delta_ELambda2 = ELambda2_Doxy - ELambda2_oxy
    
    # Calculate sO2
    numerator = M_Lambda1 * ELambda2_Doxy - M_Lambda2 * ELambda1_Doxy
    denominator = -M_Lambda2 * Delta_ELambda1 + M_Lambda1 * Delta_ELambda2
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    
    sO2 = numerator / denominator
    
    return sO2

ELambda1_oxy = 1058
ELambda1_Doxy = 691.32
ELambda2_oxy = 518
ELambda2_Doxy = 1408.24

######## Our model
model_path = r"CONFIGUN_W1.hdf5"
model_path2 = r"CONFIGUN_W2.hdf5"

model = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})
model2 = tf.keras.models.load_model(model_path2, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})

den_data = model.predict([law1l, law2l])
den_data[den_data<0.15] = 0
np.save("Tube_ours_W1", den_data)

den_data2 = model2.predict([law2l, law1l])
den_data2[den_data2<0.1] = 0
np.save("Tube_ours_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_Our_Model_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')

######## Dense_MSE
model_path = r"DenseUN_model_with_US_6_epch_30_lr_5e3_MSE_loss_wave1_intermingled_att_res.hdf5"
model_path2 = r"DenseUN_model_with_US_6_epch_30_lr_5e3_MSE_loss_wave2_intermingled_att_res.hdf5"

model = tf.keras.models.load_model(model_path, custom_objects={'intensity_aware_loss': intensity_aware_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})
model2 = tf.keras.models.load_model(model_path2, custom_objects={'intensity_aware_loss': intensity_aware_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})

den_data = model.predict([law1l])
den_data[den_data<0.15] = 0
np.save("Tube_Dense_MSE_W1", den_data)

den_data2 = model2.predict([law2l])
den_data2[den_data2<0.1] = 0
np.save("Tube_Dense_MSE_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_Dense_MSE_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')

######## Dense w/o
model_path = r"Dense_UNet_wo_US_w1_total_loss.hdf5"
model_path2 = r"Dense_UNet_wo_US_w2_total_loss.hdf5"

model = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})
model2 = tf.keras.models.load_model(model_path2, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})

den_data = model.predict([law1l])
den_data[den_data<0.15] = 0
np.save("Tube_Dense_wo_W1", den_data)

den_data2 = model2.predict([law2l])
den_data2[den_data2<0.1] = 0
np.save("Tube_Dense_wo_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_Dense_wo_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')

######## Attention-Res
model_path = r"Attention_ResUNet_wo_US_w1_total_loss.hdf5"
model_path2 = r"Attention_ResUNet_wo_US_w2_total_loss.hdf5"

model = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})
model2 = tf.keras.models.load_model(model_path2, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})

den_data = model.predict([law1l])
den_data[den_data<0.15] = 0
np.save("Tube_Attention-Res_W1", den_data)

den_data2 = model2.predict([law2l])
den_data2[den_data2<0.1] = 0
np.save("Tube_Attention-Res_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_Attention-Res_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')

######## Attention
model_path = r"Attention_UNet_wo_US_w1_total_loss.hdf5"
model_path2 = r"Attention_UNet_wo_US_w2_total_loss.hdf5"

model = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})
model2 = tf.keras.models.load_model(model_path2, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})

den_data = model.predict([law1l])
den_data[den_data<0.15] = 0
np.save("Tube_Attention_W1", den_data)

den_data2 = model2.predict([law2l])
den_data2[den_data2<0.1] = 0
np.save("Tube_Attention_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_Attention_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')

######## Res
model_path = r"ResUNet_wo_US_w1_total_loss.hdf5"
model_path2 = r"ResUNet_wo_US_w2_total_loss.hdf5"

model = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})
model2 = tf.keras.models.load_model(model_path2, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})

den_data = model.predict([law1l])
den_data[den_data<0.15] = 0
np.save("Tube_Res_W1", den_data)

den_data2 = model2.predict([law2l])
den_data2[den_data2<0.1] = 0
np.save("Tube_Res_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_Res_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')

######## UNet
model_path = r"UNet_wo_US_w1_total_loss.hdf5"
model_path2 = r"UNet_wo_US_w2_total_loss.hdf5"

model = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})
model2 = tf.keras.models.load_model(model_path2, custom_objects={'total_loss': total_loss, 'cassim':cassim, 'efd': efd, 'sif': sif, 'twiq':twiq})

den_data = model.predict([law1l])
den_data[den_data<0.02] = 0
np.save("Tube_UNet_W1", den_data)

den_data2 = model2.predict([law2l])
den_data2[den_data2<0.015] = 0
np.save("Tube_UNet_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_UNet_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')


######## cGAN
model_path = r"cGAN_model.hdf5"
model_path2 = r"cGAN_model.hdf5"

model = tf.keras.models.load_model(model_path)
model2 = tf.keras.models.load_model(model_path2)

from scipy.ndimage import zoom

law1l = zoom(law1l, (1, 2, 1, 1), order=1)
law2l = zoom(law2l, (1, 2, 1, 1), order=1)

den_data = model.predict([law1l])
den_data[den_data<0.15] = 0
den_data = zoom(den_data, (1, 0.5, 1, 1), order=1)
np.save("Tube_cGAN_W1", den_data)

den_data2 = model2.predict([law2l])
den_data2[den_data2<0.1] = 0
den_data2 = zoom(den_data2, (1, 0.5, 1, 1), order=1)
np.save("Tube_cGAN_W2", den_data2)

so2_tot = []
for chk_indx in range(0,4):
    w1 = np.squeeze(den_data[chk_indx])
    w2 = np.squeeze(den_data2[chk_indx])
    
    M_Lambda1 = w1
    M_Lambda2 = w2
    
    # Calculate sO2
    sO2 = calculate_so2(M_Lambda1, M_Lambda2, ELambda1_oxy, ELambda1_Doxy, ELambda2_oxy, ELambda2_Doxy)
    sO2 = sO2 * 100
    sO2[sO2<0] = 0
    sO2[sO2>100] = 100
    
    so2_tot.append(sO2)

savemat('Tube_cGAN_StO2.mat', {'so2_tot': so2_tot})

plt.imshow(so2_tot[0], cmap='jet')
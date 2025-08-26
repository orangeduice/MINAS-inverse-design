#!/usr/bin/env python
# coding: utf-8
"""
date: 06/2023
@author: P. Wiecha

helper functions for deep learning inverse design tutorials
"""
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_reflection_spectra_data(path_h5, save_scalers=True, test_fraction=0.05):

    with h5py.File(path_h5) as f_read:
        R_spec = np.array(f_read['R'], dtype=np.float32)
        geo = np.array(f_read['geo'], dtype=np.float32)
        wavelengths = np.array(f_read['wavelengths'], dtype=np.float32)

    geo_mask = [True, False, True, True, True]

    geo = geo[:,geo_mask]

    # clip R spectra to 64 points
    if R_spec.shape[1] > 64:
        R_spec = R_spec[:, -64:]

    if wavelengths.shape[0] > 64:
        wavelengths = wavelengths[-64:]

    # if necessary, add a channel dimension to the spectra (keras: channels last)
    if R_spec.shape[-1] != 1:
        R_spec = np.expand_dims(R_spec, -1)

    #  separately standardize permittivities and thicknesses
    scaler_geo = StandardScaler().fit(geo)

    # save the scalers using pickle
    if save_scalers:
        pickle.dump(scaler_geo,
                    open('{}_scalers.pkl'.format(os.path.splitext(path_h5)[0]), 'wb'))

    # apply scaler
    geo = scaler_geo.transform(geo)

    if geo.shape[-1] != 1:
        geo = np.expand_dims(geo, -1)

    # split into training and test datasets. Set random state for a reproducible splitting
    x_train, x_test, y_train, y_test = train_test_split(
        geo, R_spec, test_size=test_fraction, random_state=42)

    return x_train, x_test, y_train, y_test, wavelengths

def inverse_scale_geo(geo, scaler_geo):
    """mat and thick are predicted, normalized values. return their inverse transforms"""
    geo_physical = scaler_geo.inverse_transform(geo)

    return geo_physical
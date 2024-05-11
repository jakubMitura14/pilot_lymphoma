import ml_collections
import jax
import numpy as np
from ml_collections import config_dict
import more_itertools
import toolz
import einops
from jax import lax, random, numpy as jnp
import orbax.checkpoint
from datetime import datetime
from flax.training import orbax_utils
import os
import h5py
import pandas as pd
from os.path import basename, dirname, exists, isdir, join, split
import SimpleITK as sitk
import multiprocessing as mp
from functools import partial

import os
import numpy as np
import numpy as np

import SimpleITK as sitk
import jax.numpy as jnp
import itertools


def join_ct_suv(ct: sitk.Image, suv: sitk.Image,ct1: sitk.Image, suv1: sitk.Image) -> sitk.Image:
    """
    Resample a CT image to the same size as a SUV image
    """
   
    ct_arr=sitk.GetArrayFromImage(ct)
    suv_arr=sitk.GetArrayFromImage(suv)

    ct_arr_1=sitk.GetArrayFromImage(ct1)
    suv_arr_1=sitk.GetArrayFromImage(suv1)
    
    res=jnp.stack([jnp.array(suv_arr),jnp.array(ct_arr),jnp.array(ct_arr_1),jnp.array(suv_arr_1)],axis=-1)
    return res

def load_landmark_data(folder_path:str):
    """
    given path to folder with landmarks files and images after general registaration we load the data
    we want to first load the suv and ct images resample them to the same size and then load the landmarks
    we need to load separately study 0 and 1 
    the output should be in form of a dictionary with keys 'study_0','study_1','From`,`To`' where `From` and `To` are the landmarks
    all the data should be in form of jnp.arrays
    """
    ct_0=sitk.ReadImage(folder_path+'/study_0_ct_soft.nii.gz')
    suv_0=sitk.ReadImage(folder_path+'/study_0_SUVS.nii.gz')
    # Resample ct_0 to match ct_1
            
    ct_1=sitk.ReadImage(folder_path+'/study_1_ct_soft.nii.gz')
    suv_1=sitk.ReadImage(folder_path+'/study_1_SUVS.nii.gz')    
    arr_0 = join_ct_suv(ct_0, suv_0,ct_1, suv_1)

    return {'study':arr_0, 'From':jnp.load(folder_path+'/From.npy'),'To':jnp.load(folder_path+'/To.npy')}


def reshape_image(arr, img_size):
    # Get the current shape of the input array
    img_size=(img_size[1],img_size[2],img_size[3],img_size[4])
    current_shape = arr.shape
    
    # Check if the current shape is already equal to the desired shape
    if current_shape == img_size:
        print("The input array already has the desired shape.")
        return arr
    
    # Check if the current shape is larger than the desired shape in any dimension
    if any(cs > ds for cs, ds in zip(current_shape, img_size)):
        # Crop the input array from the end of the dimension where it occurs
        arr = arr[:img_size[0], :img_size[1], :img_size[2], :img_size[3]]
        print("The input array has been cropped to the desired shape.")
    
    # Check if the current shape is smaller than the desired shape in any dimension
    if any(cs < ds for cs, ds in zip(current_shape, img_size)):
        # Pad the input array with zeros at the end of the dimension where it occurs

        arr = np.pad(arr, ((0, np.max(img_size[0] - current_shape[0],0)),
                                  (0, np.max(img_size[1] - current_shape[1],0)),
                                  (0, np.max(img_size[2] - current_shape[2],0)),
                                  (0, 0)), mode='constant')
        print("The input array has been padded to the desired shape.")
    
    # If none of the above conditions are met, return the input array as is
    return arr


def stack_with_pad(arr_0,arr_1):
    if arr_0.shape[0] > arr_1.shape[0]:
        pad_length = arr_0.shape[0] - arr_1.shape[0]
        padding = jnp.full((pad_length, arr_1.shape[1]), -1)
        arr_1 = jnp.concatenate((arr_1, padding), axis=0)
    elif arr_1.shape[0] > arr_0.shape[0]:
        pad_length = arr_1.shape[0] - arr_0.shape[0]
        padding = jnp.full((pad_length, arr_0.shape[1]), -1)
        arr_0 = jnp.concatenate((arr_0, padding), axis=0)
    
    return jnp.stack([arr_0, arr_1])

    
def get_batched(folder_tuple,img_size):
    folder_0=load_landmark_data(f"{folder_tuple[0]}/general_transform")
    folder_1=load_landmark_data(f"{folder_tuple[1]}/general_transform")
    arr=jnp.stack([reshape_image(folder_0['study'],img_size),reshape_image(folder_1['study'],img_size)])
    From=stack_with_pad(folder_0['From'],folder_1['From'])
    To=stack_with_pad(folder_0['To'],folder_1['To'])
    return {'study':arr, 'From':From,'To':To}

def get_dataset(folder_path,img_size):
    folder_names = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    folder_names= list(filter(lambda el: "pat" in el, folder_names))
    folder_tuples = list(itertools.zip_longest(*[iter(folder_names)] * 2))
    folder_tuples=folder_tuples[0:19]
    return [get_batched(folder_tuple,img_size) for folder_tuple in folder_tuples]


# import jax; print(jax.devices())
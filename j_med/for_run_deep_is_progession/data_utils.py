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
    
    # res=jnp.stack([jnp.array(ct_arr),jnp.array(ct_arr_1)],axis=-1)
    res=jnp.stack([jnp.array(suv_arr),jnp.array(ct_arr),jnp.array(suv_arr_1),jnp.array(ct_arr_1)],axis=-1)
    return res

def load_landmark_data(folder_path:str):
    """
    given path to folder with landmarks files and images after general registaration we load the data
    we want to first load the suv and ct images resample them to the same size and then load the landmarks
    we need to load separately study 0 and 1 
    the output should be in form of a dictionary with keys 'study_0','study_1','From`,`To`' where `From` and `To` are the landmarks
    all the data should be in form of jnp.arrays
    """
    # ct_0=sitk.ReadImage(folder_path+'/study_0_ct_soft.nii.gz')
    ct_0=sitk.ReadImage(folder_path+'/study_0_tmtvNet_SEG.nii.gz')
    suv_0=sitk.ReadImage(folder_path+'/study_0_SUVS.nii.gz')
    # Resample ct_0 to match ct_1
            
    ct_1=sitk.ReadImage(folder_path+'/study_1_tmtvNet_SEG.nii.gz')
    # ct_1=sitk.ReadImage(folder_path+'/study_1_ct_soft.nii.gz')
    suv_1=sitk.ReadImage(folder_path+'/study_1_SUVS.nii.gz')    
    arr_0 = join_ct_suv(ct_0, suv_0,ct_1, suv_1)
    # print(f"in load_landmark_data arr_0 {arr_0.shape}  ")

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
        
        a=current_shape[0]-img_size[0]
        b=current_shape[1]-img_size[1]
        c=current_shape[2]-img_size[2]
        # print(f"aaa {a} a//2 {a//2} a-(a//2 {a-(a//2)}")
        # print(f"aaa2 {b} a//2 {b//2} a-(a//2 {b-(b//2)}")
        # print(f"aaa3 {c} a//2 {c//2} a-(a//2 {c-(c//2)}")
                    
        if a>0:
            arr = arr[a//2:-(a-(a//2)), :, :, :]
        if b>0:
            arr = arr[:, b//2:-(b-(b//2)), :, :]        
        if c>0:
            arr = arr[:, :,c//2:-(c-(c//2)), :]        
        
        print(f"The input array has been cropped to the desired shape. {arr.shape}  img_size {img_size} old shape {current_shape}")
    current_shape = arr.shape
    # Check if the current shape is smaller than the desired shape in any dimension
    if any(cs < ds for cs, ds in zip(current_shape, img_size)):
        # Pad the input array with zeros at the end of the dimension where it occurs
        # print(f"aaaaa {arr.shape}  {img_size} {current_shape}  {np.max(img_size[0] - current_shape[0],0)}  {np.max(img_size[1] - current_shape[1],0)}  {np.max(img_size[2] - current_shape[2],0)}")
        arr = np.pad(arr, ((0, np.max(img_size[0] - current_shape[0],0)),
                                  (0, np.max(img_size[1] - current_shape[1],0)),
                                  (0, np.max(img_size[2] - current_shape[2],0)),
                                  (0, 0)), mode='constant')
        print(f"The input array has been padded to the desired shape. {arr.shape}  {img_size} ")
    

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

def get_pat_num_from_path(p):
    return int(p.split("/")[-1].split("_")[1])
    
    
def get_not_batched(folder_name,img_size):
    # folder_0=load_landmark_data(f"{folder_name}/lin_transf")
    folder_0=load_landmark_data(f"{folder_name}/general_transform")

    arrr_1=reshape_image(folder_0['study'],img_size)
    # arrr_1=reshape_image(jnp.expand_dims(folder_0['study'],axis=0),img_size)
    # print(f"  arrr_1 {arrr_1.shape}  arrr_2 {arrr_2.shape}  {img_size}")
    # arr=jnp.stack([arrr_1])

    
    full_data_table_path="/workspaces/pilot_lymphoma/data/full_table_data_for_delta.csv"
    full_data_table= pd.read_csv(full_data_table_path)
    full_data_table["pat_id"]=full_data_table["Unnamed: 0"].astype(int)
    full_data_table["outcome"]=full_data_table["Unnamed: 12"]

    outcome_dict = {'CR':0, 'PD':1, 'PR':0, 'SD':0}

    outcomes_pat=list(zip(full_data_table["pat_id"].to_numpy(),full_data_table["outcome"].to_numpy()))
    outcome_dict_fin=dict(list(map(lambda pair: (pair[0],outcome_dict[pair[1]]),outcomes_pat )))

    outcomee=jnp.array([outcome_dict_fin[get_pat_num_from_path(folder_name)]])
    
    return {'study':jnp.expand_dims(arrr_1,axis=0), 'outcome':outcomee}

def get_dataset(folder_path,img_size):
    folder_names = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    folder_names= list(filter(lambda el: "pat" in el, folder_names))
    # folder_tuples = list(itertools.zip_longest(*[iter(folder_names)] * 2))
    # folder_tuples=folder_tuples[0:19]
    return [get_not_batched(folder_name,img_size) for folder_name in folder_names]


# import jax; print(jax.devices())
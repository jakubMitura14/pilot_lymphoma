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

def reg_a_to_b_by_metadata_single_b(fixed_image_path,moving_image_path):
    # moving_image_path=moving_image_path[0]
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)

    # fixed_image=sitk.Cast(fixed_image, sitk.sitkUInt8)
    # moving_image=sitk.Cast(moving_image, sitk.sitkInt)
    
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkBSpline, 0)
    return sitk.GetArrayFromImage(resampled)
    # print(f" prim sum {np.sum(sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path)).flatten())} \n suuum {np.sum(sitk.GetArrayFromImage(resampled).flatten())} ")
  
    # writer = sitk.ImageFileWriter()
    # new_path= join(out_folder,moving_image_path.split('/')[-1])
    # writer.SetFileName(new_path)
    # writer.Execute(resampled)

    # return new_path


def apply_on_single(pathh,registered_path, total_slices):    
    innerfiles=os.listdir(join(registered_path,pathh))
    transformed= list(filter(lambda el: 'trans' in el,innerfiles))[0]
    one= list(filter(lambda el: 'sudy_1' in el,innerfiles))[0]
    transformed= join(registered_path,pathh,transformed)
    one= join(registered_path,pathh,one)

    regg=reg_a_to_b_by_metadata_single_b(one,transformed)
    one= sitk.GetArrayFromImage(sitk.ReadImage(one))

    # total_slices=420
    diff= regg.shape[0]-total_slices
    diff_quart= diff/6  
    diff_end=round(diff_quart*5)
    diff_beg=regg.shape[0]-(diff-diff_end)
    regg0=regg.shape[0]
    regg=regg[diff_end:diff_beg,:,:]
    one=one[diff_end:diff_beg,:,:]
    # regg=regg[diff_beg:diff_end,:,:]
    # one=one[diff_beg:diff_end,:,:]


    # print(f"regg 0  {regg0} regg {regg.shape[0]} diff_end {diff_end} diff_beg {diff_beg}")


    return np.stack([regg,one],axis=-1)



def add_batches(patient_paths,registered_path, total_slices,labels_df_path,batch_size):
  

  batch_size_pmapped=np.max([batch_size//jax.local_device_count(),1])
  df= pd.read_csv(labels_df_path)
  labels=np.array(list(map(lambda pathh : df.loc[df['Unnamed: 0']==pathh]['deauville_2'].to_numpy()[0],patient_paths)))>3
  data=[]
  with mp.Pool(processes = mp.cpu_count()) as pool: 
    # data=pool.map(lambda pathh: apply_on_single(pathh,registered_path, total_slices)   ,patient_paths)
    data=pool.map(partial(apply_on_single,registered_path=registered_path, total_slices=total_slices)   ,patient_paths)

  data = jnp.stack(data)
  labels=jnp.array(labels)
  dv=jax.local_device_count()

  data= einops.rearrange(data,'(i pm b) x y z c-> i pm b x y z c' ,pm=dv,b=batch_size)
  labels= einops.rearrange(labels,'(i pm b)-> i pm b' ,pm=dv,b=batch_size)


  # cached_subj=list(map(apply_on_single,cached_subj ))
  # batch_images,batch_labels=list(toolz.sandbox.core.unzip(cached_subj))
  # batch_images= list(batch_images)
  # batch_labels= list(batch_labels)
  # batch_images= jnp.concatenate(batch_images,axis=0 )
  # batch_labels= jnp.concatenate(batch_labels,axis=0 ) #b y z
  # #padding to be able to use batch size efficiently
  # target_size= int(np.ceil(batch_images.shape[0]/batch_size))*batch_size
  # to_pad=target_size- batch_images.shape[0]
  # if(to_pad>0):
  #    batch_images= jnp.pad(batch_images,((0,to_pad),(0,0),(0,0)))
  #    batch_labels= jnp.pad(batch_labels,((0,to_pad),(0,0),(0,0)))
  # batch_images= einops.rearrange(batch_images,'(d pm b) x y->d pm b x y 1',b=batch_size_pmapped,pm=jax.local_device_count())  
  # batch_labels= einops.rearrange(batch_labels,'(d pm b) x y->d pm b x y 1',b=batch_size_pmapped,pm=jax.local_device_count())  
  # print(f"add_batches batch_images {batch_images.shape} batch_labels {batch_labels.shape}")
  # batch_images=batch_images[:-1,:,:,:,:,:]
  # batch_labels=batch_labels[:-1,:,:,:,:,:]
  return data,labels



# registered_path= '/workspaces/pilot_lymphoma/data/fid_registered/fiducially_registered'
# labels_df_path='/workspaces/pilot_lymphoma/data/all_deauville_anon - Copy of Form responses 1.csv'
# total_slices=420
# batch_size=1

# patient_paths=os.listdir(registered_path)
# data,labels=add_batches(patient_paths,registered_path, total_slices,labels_df_path,batch_size)
# print(f"data {data.shape}")


# def get_check_point_folder():
#   now = datetime.now()
#   checkPoint_folder=f"/workspaces/Jax_cuda_med/data/checkpoints/{now}"
#   checkPoint_folder=checkPoint_folder.replace(' ','_')
#   checkPoint_folder=checkPoint_folder.replace(':','_')
#   checkPoint_folder=checkPoint_folder.replace('.','_')
#   os.makedirs(checkPoint_folder)
#   return checkPoint_folder  
   

# def save_checkpoint(index,epoch,cfg,checkPoint_folder,state,loss):
#     if(index==0 and epoch%cfg.divisor_checkpoint==0 and cfg.to_save_check_point):
#         chechpoint_epoch_folder=f"{checkPoint_folder}/{epoch}"
#         # os.makedirs(chechpoint_epoch_folder)

#         orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#         ckpt = {'model': state, 'config': cfg,'loss':loss}
#         save_args = orbax_utils.save_args_from_target(ckpt)
#         orbax_checkpointer.save(chechpoint_epoch_folder, ckpt, save_args=save_args)

# def save_examples_to_hdf5(masks,batch_images_prim,curr_label  ):
#     f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/example_mask.hdf5', 'w')
#     f.create_dataset(f"masks",data= masks)
#     f.create_dataset(f"image",data= batch_images_prim)
#     f.create_dataset(f"label",data= curr_label)
#     f.close()   






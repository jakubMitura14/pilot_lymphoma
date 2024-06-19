from matplotlib.pylab import *
from jax import  numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
# import monai_swin_nD
import tensorflow as tf
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
import h5py
import jax
from ml_collections import config_dict
from skimage.segmentation import mark_boundaries
import cv2
import functools
import flax.jax_utils as jax_utils
import tensorflow as tf
from jax_smi import initialise_tracking
import ml_collections
import time
import more_itertools
import toolz
from subprocess import Popen
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint
from datetime import datetime
from flax.training import orbax_utils
from flax.core.frozen_dict import freeze
import flax
import jax_metrics as jm
import functools
import re
import typing
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jax import config
from ..testUtils.tensorboard_utils import *
# import augmentations.simpleTransforms
# from augmentations.simpleTransforms import main_augment
import numpy as np
from monai.transforms import Affined, Compose
from monai.utils import ensure_tuple_rep

# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms.functional as F
# import torchvision

from .geometric_sv_model import Pilot_modell
from .xla_utils import *
from flax.training import checkpoints

from sklearn.model_selection import KFold
import numpy as np
from monai.transforms import (
    Compose,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandRicianNoised,
    Rand3DElasticd
)




SCRIPT_DIR = '/root/externalRepos/big_vision'
sys.path.append(str(SCRIPT_DIR))

# from big_vision import optax as bv_optax
# from big_vision.pp import builder as pp_builder
# from big_vision.trainers.proj.gsam.gsam import gsam_gradient

from .config_out_image import get_cfg
from .tensorboard_for_out_image import *
from .data_utils import *
import os
import sys
import pathlib
from jax import random

config.update("jax_debug_nans", True)
# Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], 'GPU')


#get configuration
cfg= get_cfg()
file_writer=setup_tensorboard()

def rotate_and_translate(points: jnp.ndarray
                         , center_point: jnp.ndarray, rotation_vector: jnp.ndarray,
                         translation_vector: jnp.ndarray) -> jnp.ndarray:
    # rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()

    # rotation_matrix =jnp.linalg.inv(Rotation.from_euler('xyz', [0.0,0.0,0.0], degrees=True).as_matrix())
    # return ((points - center_point) @ rotation_matrix.T + center_point) 
    rotation_matrix =jnp.linalg.inv(Rotation.from_euler('xyz', rotation_vector, degrees=True).as_matrix())
    return ((points - center_point) @ rotation_matrix.T + center_point) + translation_vector





def initt(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model):
  """Creates initial `TrainState`."""
  img_size=list(cfg.img_size)
  # img_size[0]=img_size[0]//jax.local_device_count()
  input=jnp.ones((img_size[1],img_size[2],img_size[3],img_size[4]))
  rng_main,rng_mean=jax.random.split(rng_2)
  print(f"iiiiiiiiiiiii init {input.shape}")
  #jax.random.split(rng_2,num=1 )
  params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  # cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  # # decay_scheduler=optax.linear_schedule(cfg.learning_rate, cfg.learning_rate/10, cfg.total_steps, transition_begin=0)
  # sgdr_schedulee=optax.sgdr_schedule([cfg.learning_rate])
  # joined_scheduler=optax.join_schedules([optax.constant_schedule(cfg.learning_rate*10),optax.constant_schedule(cfg.learning_rate)], [10])


  tx = optax.chain(
        optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
        # optax.lion(learning_rate=joined_scheduler)
        # optax.lion(learning_rate=cfg.learning_rate)
        #optax.lion(learning_rate=decay_scheduler)
        # optax.fromage(learning_rate=0.003)
        optax.nadamw(learning_rate=cfg.learning_rate)
        
        )
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def update_fn(state, image,outcome, cfg,model):

  """Train for a single step."""
  def loss_fn(params,image,outcome):
    conved=model.apply({'params': params}, image, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
    # print(f"aaaaaaaaaaaaa conved {conved.shape} sum {jnp.sum(conved)} image {image.shape} outcome {outcome.shape}")
    one_hot=nn.one_hot(jnp.array(outcome).astype(int),cfg.num_classes).astype(float)
    loss=optax.losses.sigmoid_binary_cross_entropy(jnp.expand_dims(conved,axis=0),jnp.expand_dims(one_hot,axis=0))
    return jnp.mean(loss)

  grad_fn = jax.value_and_grad(loss_fn)
  l, grads = grad_fn(state.params,image,outcome)
  state=state.apply_gradients(grads=grads)

  return state,l






def simple_apply(state, image,outcome, cfg,model,metric):
  weights=model.apply({'params': state.params}, image, rngs={'to_shuffle': random.PRNGKey(2)}).flatten()#, rngs={'texture': random.PRNGKey(2)}
  weights=nn.sigmoid(weights)
  weights=jnp.round(weights)
  
  one_hot=nn.one_hot(jnp.array(outcome).astype(int),cfg.num_classes)
  # metricc = metric.update(target=jnp.expand_dims(one_hot,axis=0).astype(int), preds=jnp.expand_dims(weights,axis=0))
  # metric_inner = jm.metrics.Accuracy()
  # metric_inner = metric_inner.update(target=jnp.expand_dims(one_hot,axis=0).astype(int), preds=jnp.expand_dims(weights,axis=0))
  metric_inner=(weights[1]==outcome[0]).astype(float)
  print(f"in simple apply  {weights}  outcome {outcome} metricc {metric_inner}")
  return metric_inner



def train_epoch(batch_images,outcome,epoch,index
                ,model,cfg
                ,rng_loop
                ,state
                ,is_pretraining
                ,metric
                ):    
  epoch_loss=[]


  state,loss=update_fn(state, batch_images, outcome,cfg,model)
  epoch_loss.append(jnp.mean(loss.flatten())) 
  metric=simple_apply(state, batch_images, outcome,cfg,model,metric)

  # print(f"metr {np.mean(metr)}")
  # with file_writer.as_default():
  #     tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
  #     tf.summary.scalar(f"metr ", np.mean(metr),       step=epoch)
  return state,loss,metric



def main_train(cfg):

  prng = jax.random.PRNGKey(42)
  model = Pilot_modell(cfg)
  rng_2=jax.random.split(prng,num=1)
  # batch_size=2
  img_size = cfg.img_size 
  folder_path='/root/data/prepared_registered'
  checkpoints_fold="/workspaces/pilot_lymphoma/data/fiducial_model_checkPoints"
  dataset=get_dataset(folder_path,img_size)
  # dataset_a=list(map(lambda el: {'study': jnp.expand_dims(el['study'][0,:,:,:,:],0),'outcome' :jnp.expand_dims(el['outcome'][0,:],0)} , dataset))
  # dataset_b=list(map(lambda el: {'study': jnp.expand_dims(el['study'][1,:,:,:,:],0),'outcome' :jnp.expand_dims(el['outcome'][1,:],0)}, dataset))
  # dataset=dataset_a+dataset_b
  # print(f"rrrrrrrrrrrr {dataset_b[1]['study'].shape}  From {dataset_b[1]['From'].shape}")
  # {'study':arr, 'From':From,'To':To}
  
  dataset_len=len(dataset)
  dataset_indicies=np.arange(0,dataset_len)
  

  # Create a KFold object
  kf = KFold(n_splits=5)

  # Use the KFold object to generate the training and validation sets
  fold_index=0
  for train_index, val_index in kf.split(dataset_indicies):
    fold_index=fold_index+1
    train_set = dataset_indicies[train_index]
    val_set = dataset_indicies[val_index]
    print(f"train_index {fold_index} train_set {train_set} val_set {val_set}")



  
  
    state= initt(prng,cfg,model)  
    metric = jm.metrics.Accuracy()
    metric_val = jm.metrics.Accuracy()

  


    dataset_curr=[dataset[i] for i in train_set]
    dataset_curr_val=[dataset[i] for i in val_set]
    for epoch in range(1, cfg.total_steps):
        is_pretraining=False
        if(epoch<500):
          is_pretraining=True
        prng, rng_loop = jax.random.split(prng, 2)
        metres=[]
        metres_val=[]
        losses=[]
        for index in range(len(dataset_curr)) :
          curr_data=dataset_curr[index]
          # print(f"epoch {epoch} index {index}")

          # Define the augmentation transform
          # augmentation = tio.Compose([
          #   # tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=10),
          #   tio.RandomElasticDeformation(num_control_points=7, max_displacement=7),
          #   tio.RandomNoise(std=(0, 0.1)),
          # ])
          # Apply the augmentation to the current study
          imm_now=curr_data["study"]
          imm_now_shape=imm_now.shape
          # print(f"iiiiiiiiiii imm_now_shape {imm_now_shape}")
          ### apply thhose transforms always  
          transform = Compose([
            # RandShiftIntensityd(keys="img",offsets=1.0,prob=0.9),
            # RandScaleIntensityd(keys="img",factors=1.0,prob=0.9),
            Rand3DElasticd(keys="img",sigma_range=(5, 8), magnitude_range=(100, 200),prob=0.2),
            # RandAdjustContrastd(keys="img"),
            # RandGaussianSmoothd(keys="img"),
            # RandGaussianSharpend(keys="img"),
            RandRicianNoised(keys="img",prob=0.2),
            
          ])


          # Apply the transform
          imm_now=np.array(einops.rearrange(imm_now,'b h w d c-> b c h w d'))

          # with mp.Pool(2) as p:
          imm_now = transform({"img": imm_now[0,:,:,:,:]})["img"].cpu().numpy()
          imm_now=jnp.stack([imm_now])
          imm_now=einops.rearrange(imm_now,'b c h d w -> b h w d c')
            
    
      
            # Update the current study with the augmented study
          state,loss,metricc=train_epoch(imm_now[0,:,:,:,:],curr_data["deauville"],epoch,index
                                          ,model,cfg
                                          ,rng_loop,
                                          state,is_pretraining,metric)
          
          
          metres.append(metricc)
          
          losses.append(loss)
          
        # checkpoints.save_checkpoint(f"{checkpoints_fold}/{jnp.mean(metr)}__{epoch}", jax_utils.unreplicate(state), step=step)
        # shutil.rmtree(f"{checkpoints_fold}/now", ignore_errors=True)
        # checkpoints.save_checkpoint(f"{checkpoints_fold}/now", state, step=step)
        
        for index in range(len(dataset_curr_val)) :
          curr_data_val=dataset_curr_val[index]
          metric_vall=simple_apply(state, curr_data_val["study"],curr_data_val["deauville"], cfg,model,metric_val)
          metres_val.append(metric_vall)
        print(f"******* epoch {epoch} ")
        print(f"loss {np.mean(np.mean(losses))} ")
        #  metres=[]
        # metres_val=[]       
        # acc = metric.compute()   
        # acc_val = metric_val.compute()   
        # metric.reset()
        # metric_val.reset()
        
        print(f"metr {np.mean(np.mean(metres))} ")
        print(f"metr val {np.mean(np.mean(metres_val))} ")

        with file_writer.as_default():
            # print(f"per_corr {per_corr} corr_total {corr_total} incorr_total {incorr_total}")
            tf.summary.scalar(f"f{fold_index}_loss", np.mean(np.mean(losses)) ,       step=epoch)
            tf.summary.scalar(f"f{fold_index}_acc_val", np.mean(np.mean(metres_val)) ,       step=epoch)
            tf.summary.scalar(f"f{fold_index}_acc", np.mean(np.mean(metres)) ,       step=epoch)
            # tf.summary.scalar(f"f{fold_index}_metr_val", np.mean(np.mean(metres_val)) ,       step=epoch)


# Reset the metric
main_train(cfg)


# tensorboard --logdir=/workspaces/pilot_lymphoma/data/tensor_board

# python3 -m j_med.for_run_deep_dauville.run_pilot_lymphoma_single_gpu



# None [327.3,413, 479.3,403, 25.3 ] mean 329.58
# lin transf [18.4, 20.1,19.5,14.5,13.1 ] mean 329.58
# in case pure general transform -[132.93,158.90,62.06,580.9,346.61] mean 256.28
# result from None transformation on the end of 800 epoch training [70,474,489,486,356] #375


#lin registered is progression accuracy 0.833, 0.667, 0.833, 0.6, 0.8
#general transform is progression accuracy 0.833, 0.667, 0.833, 0.6, 0.8

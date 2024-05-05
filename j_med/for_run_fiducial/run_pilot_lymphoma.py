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
from jax.config import config
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

from ..testUtils.tensorboard_utils import *
# import augmentations.simpleTransforms
# from augmentations.simpleTransforms import main_augment

# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms.functional as F
# import torchvision

from .geometric_sv_model import Pilot_model




SCRIPT_DIR = '/root/externalRepos/big_vision'
sys.path.append(str(SCRIPT_DIR))

from big_vision import optax as bv_optax
from big_vision.pp import builder as pp_builder
from big_vision.trainers.proj.gsam.gsam import gsam_gradient

from .config_out_image import get_cfg
from .tensorboard_for_out_image import *
from .data_utils import *
import os
import sys
import pathlib

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
    rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()
    return (points - center_point) @ rotation_matrix.T + center_point + translation_vector


def get_fiducial_loss(weights,from_landmarsk,to_landmarks,image_shape):
    """
    first entries in in weights are :
        0-3 first rotation vector   
        3-6 translation vector   
    so we interpret the weights as input for transormations rotation;translation
    we apply this transformation to the fiducial points of moving image and 
    calculate the square distance between transformed fiducial points and fiducial points on fixed image           
    """
    center_point=(jnp.asarray(image_shape) - 1.) / 2
    res=rotate_and_translate(to_landmarks, center_point, weights[0:3],weights[3:6])
    #calculate the square distance between transformed fiducial points and fiducial points on fixed image 
    return jnp.sum(((from_landmarsk-res)**2).flatten())
    
    
def transform_image(image,weights):
    """
    first entries in in weights are :
    0-3 first rotation vector   
    3-6 translation vector   
    6-9 second rotation vector 
    so we interpret the weights as input for transormations rotation;translation;rotation    
    """    
    r = Rotation.from_rotvec(jnp.array([0.1, 0.2, 0.3]))
    # imagee= jnp.zeros((3,3,3))



    imagee=r.apply(imagee)
    imagee=jax.image.scale_and_translate(imagee, imagee.shape,jnp.array([0,1,2]), jnp.array([1.0,1.0,1.0]), jnp.array([0.1,0.0,0.0]), "bicubic")
    return imagee


@functools.partial(jax.pmap,static_broadcasted_argnums=(1,2), axis_name='ensemble')#,static_broadcasted_argnums=(2)
def initt(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model):
  """Creates initial `TrainState`."""
  img_size=list(cfg.img_size)
  img_size[0]=img_size[0]//jax.local_device_count()
  input=jnp.ones(tuple(img_size))
  rng_main,rng_mean=jax.random.split(rng_2)

  #jax.random.split(rng_2,num=1 )
  params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  # cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  decay_scheduler=optax.linear_schedule(cfg.learning_rate, cfg.learning_rate/10, cfg.total_steps, transition_begin=0)
  
  joined_scheduler=optax.join_schedules([optax.constant_schedule(cfg.learning_rate*10),optax.constant_schedule(cfg.learning_rate)], [10])


  tx = optax.chain(
        optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
        # optax.lion(learning_rate=joined_scheduler)
        # optax.lion(learning_rate=cfg.learning_rate)
        optax.lion(learning_rate=decay_scheduler)
        # optax.adafactor()
        
        )
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)




@partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(4,5))
def update_fn(state, image,from_landmarsk,to_landmarks, cfg,model):
  
  """Train for a single step."""
  def loss_fn(params,image,from_landmarsk,to_landmarks):
    conved=model.apply({'params': params}, image, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
    print(f"ccccccccccccc {conved.shape} from_landmarsk {from_landmarsk.shape} to_landmarks {to_landmarks.shape}")
    loss=get_fiducial_loss(conved.flatten(),from_landmarsk,to_landmarks,(cfg.img_size[2],cfg.img_size[3],cfg.img_size[4]))
    # loss=optax.sigmoid_binary_cross_entropy( conved.flatten() ,booll_label.flatten())
    # print(f"booll_label {booll_label.shape} conved {conved.shape}")
    return jnp.mean(loss)

  grad_fn = jax.value_and_grad(loss_fn)
  l, grads = grad_fn(state.params,image,from_landmarsk,to_landmarks)
  state=state.apply_gradients(grads=grads)

  return state,l




# @partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(3))
# def simple_apply(state, image,labels,model):
#   conved=model.apply({'params': state.params}, image,rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
#   conved= nn.sigmoid(conved)
#   conved_orig=conved
#   conved=jnp.round(conved).astype(bool)
#   # batch_updates = metric.batch_updates(target=labels, preds=nn.sigmoid(conved))
#   # batch_updates = metric.update(target=labeels, preds=nn.sigmoid(conved))
#   correct= jnp.equal(conved.flatten(),labels.flatten())
#   return conved,conved_orig,correct



def train_epoch(batch_images,from_landmarsk,to_landmarks,epoch,index
                ,model,cfg
                ,rng_loop
                ,state
                ):    
  epoch_loss=[]

  state,loss=update_fn(state, batch_images, from_landmarsk,to_landmarks,cfg,model)
  epoch_loss.append(jnp.mean(jax_utils.unreplicate(loss).flatten())) 

  with file_writer.as_default():
      tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
  return state,loss




def main_train(cfg):

  prng = jax.random.PRNGKey(42)
  model = Pilot_model(cfg)
  rng_2=jax.random.split(prng,num=jax.local_device_count() )
  registered_path= '/workspaces/pilot_lymphoma/data/fid_registered/fiducially_registered'
  labels_df_path='/workspaces/pilot_lymphoma/data/all_deauville_anon - Copy of Form responses 1.csv'
  batch_size=cfg.batch_size_pmapped

  patient_paths=os.listdir(registered_path)
  patient_paths=patient_paths#[0:8]#TODO remove
  data,labels=add_batches(patient_paths,registered_path,labels_df_path,batch_size)
  print(f"data ready data {data.shape}")
  state= initt(rng_2,cfg,model)  
  # metric = jm.metrics.Accuracy()

 



  for epoch in range(1, cfg.total_steps):
      prng, rng_loop = jax.random.split(prng, 2)
      corr_total=0
      incorr_total=0

      corr_total_test=0
      incorr_total_test=0

      for index in range(data.shape[0]-2) :
        print(f"epoch {epoch} index {index}")
        state,loss=train_epoch(data[index,:,:,:,:,:,:],labels[index,:,:],epoch,index
                                         #,tx, sched_fns,params_cpu
                                         ,model,cfg
                                         ,rng_loop,
                                        #  ,params_repl, opt_repl
                                         state)


      print(f"loss {loss} ")
      # acc = metric.compute()   
      with file_writer.as_default():
          # print(f"per_corr {per_corr} corr_total {corr_total} incorr_total {incorr_total}")
          tf.summary.scalar(f"loss", loss ,       step=epoch)


      # Reset the metric



main_train(cfg)


# tensorboard --logdir=/workspaces/pilot_lymphoma/data/tensor_board

# python3 -m j_med.for_run.run_pilot_lymphoma

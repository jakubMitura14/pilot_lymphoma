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

# @partial(jax.jit, backend="cpu",static_argnums=(1,2,3))
# @functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
# def initt(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
#   img_size=list(cfg.img_size)
#   img_size[0]=img_size[0]//jax.local_device_count()
#   rng,rng_mean=jax.random.split(rng_2)
#   dummy_input = jnp.zeros(img_size, jnp.float32)
#   # params = flax.core.unfreeze(model.init(rng, dummy_input,dynamic_cfg))["params"]  
#   params = model.init({'params': rng,'to_shuffle':rng_mean  }, dummy_input,dynamic_cfg)['params'] 

#   cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   tx = optax.chain(
#         optax.clip_by_global_norm(6.0),  # Clip gradients at norm 
#         optax.lion(learning_rate=cfg.learning_rate))

#   return train_state.TrainState.create(
#       apply_fn=model.apply, params=params, tx=tx)



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
  # print(f"ppppppppparams  {params}")
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)




@partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(3,4))
def update_fn(state, image,booll_label, cfg,model):
  
  """Train for a single step."""
  def loss_fn(params,image,booll_label):
    conved=model.apply({'params': params}, image, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
    loss=optax.sigmoid_binary_cross_entropy( conved.flatten() ,booll_label.flatten())
    # print(f"booll_label {booll_label.shape} conved {conved.shape}")
    return jnp.mean(loss)

  # learning_rate = sched_fn(step) * cfg.lr
  # l=None
  # grads=None
  # if(cfg.is_gsam):

  # l, grads = gsam_gradient(loss_fn=loss_fn, params=state.params, inputs=image,
  #     targets=booll_label, lr=cfg.lr, **cfg.gsam)
  # l, grads = jax.lax.pmean((l, grads), axis_name="batch")    

  # else:
  grad_fn = jax.value_and_grad(loss_fn)
  l, grads = grad_fn(state.params,image,booll_label)
  state=state.apply_gradients(grads=grads)

  # state = update_model(state, grads)

  # l = jax.lax.pmean((l), axis_name="batch")


  # updates, opt = tx.update(grads, opt, params)
  # params = optax.apply_updates(params, updates)
  # gs = jax.tree_leaves(bv_optax.replace_frozen(cfg.schedule, grads, 0.))
  # measurements["l2_grads"] = jnp.sqrt(sum(jnp.vdot(g, g) for g in gs))
  # ps = jax.tree_util.tree_leaves(params)
  # measurements["l2_params"] = jnp.sqrt(sum(jnp.vdot(p, p) for p in ps))
  # us = jax.tree_util.tree_leaves(updates)
  # measurements["l2_updates"] = jnp.sqrt(sum(jnp.vdot(u, u) for u in us))

  return state,l




@partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(3))
def simple_apply(state, image,labels,model):
  conved=model.apply({'params': state.params}, image,rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
  conved= nn.sigmoid(conved)
  conved_orig=conved
  conved=jnp.round(conved).astype(bool)
  # batch_updates = metric.batch_updates(target=labels, preds=nn.sigmoid(conved))
  # batch_updates = metric.update(target=labeels, preds=nn.sigmoid(conved))
  correct= jnp.equal(conved.flatten(),labels.flatten())
  return conved,conved_orig,correct



def train_epoch(batch_images,booll_label,epoch,index
                # ,tx, sched_fns,params_cpu
                ,model,cfg
                ,rng_loop
                ,state
                ):    
  epoch_loss=[]
  # rngs_loop = flax.jax_utils.replicate(rng_loop)
  # print(f"state {state[1]}")
  state,loss=update_fn(state, batch_images, booll_label,cfg,model)
  epoch_loss.append(jnp.mean(jax_utils.unreplicate(loss).flatten())) 

  # #if indicated in configuration will save the parameters
  # if(index==0):
  #   save_checkpoint(index,epoch,cfg,checkPoint_folder,state,np.mean(epoch_loss))
  

  # if(index==0 and epoch%cfg.divisor_logging==0):
  #   # # losses,masks,out_image=model.apply({'params': state.params}, batch_images[0,:,:,:,:],dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
  #   # losses,masks=simple_apply(state, batch_images, dynamic_cfg,cfg,index,model)
  #   # #overwriting masks each time and saving for some tests and debugging
  #   # save_examples_to_hdf5(masks,batch_images_prim,curr_label)
  #   # #saving images for monitoring ...
  #   # mask_0=save_images(batch_images_prim,slicee,cfg,epoch,file_writer,curr_label,masks)
  #   with file_writer.as_default():
  #       tf.summary.scalar(f"mask_0 mean", np.mean(mask_0.flatten()), step=epoch)    

  with file_writer.as_default():
      tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
  # print(f"losss {np.mean(epoch_loss)}")       
  return state,loss




def main_train(cfg):
  slicee=57#57 was nice
  # checkpoint_path='/workspaces/Jax_cuda_med/data/checkpoints/2023-06-12_06_21_11_143817/1755'
  # checkpoint_path='/workspaces/Jax_cuda_med/data/checkpoints/2023-06-14_15_53_12_704500/375'
  # checkpoint_path='/workspaces/Jax_cuda_med/data/checkpoints/2023-06-29_19_23_15_314632/180'
  prng = jax.random.PRNGKey(42)
  model = Pilot_model(cfg)
  rng_2=jax.random.split(prng,num=jax.local_device_count() )
  registered_path= '/workspaces/pilot_lymphoma/data/fid_registered/fiducially_registered'
  labels_df_path='/workspaces/pilot_lymphoma/data/all_deauville_anon - Copy of Form responses 1.csv'
  total_slices=420
  batch_size=cfg.batch_size_pmapped

  patient_paths=os.listdir(registered_path)
  patient_paths=patient_paths#[0:8]#TODO remove
  data,labels=add_batches(patient_paths,registered_path, total_slices,labels_df_path,batch_size)
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
      # Get the current value of the metric
      # update metric
        # conved=model.apply({'params': state.params}, data[index,:,:,:,:,:,:], rngs={'to_shuffle': random.PRNGKey(2)})
        conved,conved_orig,correct=simple_apply(state, data[index,:,:,:,:,:,:],labels[index,:,:],model)
        # corr_total=corr_total+jnp.sum(corr.flatten())
        # incorr_total=incorr_total+ corr.flatten().shape[0] -jnp.sum(corr.flatten())
        #print(f"conved {conved[0]} gold {labels[index,0,0]} conved_orig {conved_orig} correct {correct}")
        corr_total=corr_total+np.sum(correct)
        incorr_total=incorr_total+(conved.flatten().shape[0]-np.sum(correct))
        # gather over all devices and reduce
        # batch_updates = jax.lax.all_gather(batch_updates, "device").reduce()
        # update metric
        # metric = metric.merge(batch_updates)

      conved,conved_orig,correct=simple_apply(state, data[-1,:,:,:,:,:,:],labels[-1,:,:],model)
      corr_total_test=corr_total_test+np.sum(correct)
      incorr_total_test=incorr_total_test+(conved.flatten().shape[0]-np.sum(correct))
    
      conved,conved_orig,correct=simple_apply(state, data[-2,:,:,:,:,:,:],labels[-2,:,:],model)
      corr_total_test=corr_total_test+np.sum(correct)
      incorr_total_test=incorr_total_test+(conved.flatten().shape[0]-np.sum(correct))

      print(f"corr_total_test {corr_total_test}  incorr_total_test {incorr_total_test}")
      # acc = metric.compute()   
      with file_writer.as_default():
          per_corr=corr_total/(corr_total+incorr_total)
          per_corr_test=corr_total_test/(corr_total_test+incorr_total_test)
          # print(f"per_corr {per_corr} corr_total {corr_total} incorr_total {incorr_total}")
          tf.summary.scalar(f"percent correct ", per_corr ,       step=epoch)
          tf.summary.scalar(f"percent correct test ", per_corr_test ,       step=epoch)


      # Reset the metric

# jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
# tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board

# cmd_terminal=f"tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board"
# p = Popen(cmd_terminal, shell=True)
# p.wait(5)

# jax.profiler.start_server(9999)
# logdir="/workspaces/Jax_cuda_med/data/tensor_board"
# tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

# jax.profiler.start_trace(logdir)
# with jax.profiler.trace("/workspaces/Jax_cuda_med/data/profiler_data", create_perfetto_link=True):
tic_loop = time.perf_counter()

main_train(cfg)

x = random.uniform(random.PRNGKey(0), (100, 100))
jnp.dot(x, x).block_until_ready() 
toc_loop = time.perf_counter()
print(f"loop {toc_loop - tic_loop:0.4f} seconds")

# jax.profiler.stop_trace()


# with jax.profiler.trace("/workspaces/Jax_cuda_med/data/profiler_data", create_perfetto_link=True):
#   x = random.uniform(random.PRNGKey(0), (100, 100))
#   jnp.dot(x, x).block_until_ready() 
# orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
# raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/2023-04-22_14_01_10_321058/41')
# raw_restored['model']['params']

# tensorboard --logdir=/workspaces/pilot_lymphoma/data/tensor_board

# python3 -m j_med.for_run.run_pilot_lymphoma

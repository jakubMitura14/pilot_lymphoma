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
import tensorflow as tf
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  
from torch.utils.data import DataLoader
import jax.profiler
import ml_collections
from ml_collections import config_dict
# from Jax_cuda_med.super_voxels.SIN.SIN_jax.model_sin_jax_utils import *
import pandas as pd


class Conv_trio(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    channels: int
    strides:Tuple[int]=(1,1,1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=nn.Conv(self.channels, kernel_size=(3,3,3),strides=self.strides)(x)
        # x=nn.LayerNorm()(x)
        return jax.nn.gelu(x)

def collapse_dense(conved):
    return nn.Sequential([
        # nn.Dense(128) #Initializer expected to generate shape (128, 128) but got shape (16, 128)
        nn.Dense(1200)
        ,jax.nn.gelu
        ,nn.Dense(800)          
        # ,jax.nn.gelu
        ,nn.Dense(2)
        # ,nn.sigmoid
    ])(conved.flatten())


v_collapse_dense=jax.vmap(collapse_dense,in_axes=(0))

class Pilot_modell(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    
    def setup(self):
        pass


    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        conved=nn.Sequential([
             Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2,2))
            ,nn.LayerNorm()
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2,2))
            ,nn.LayerNorm()
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2,2))
            ,nn.LayerNorm()
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2,2))
            ,nn.LayerNorm()
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2,2))
            ,nn.LayerNorm()
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2,2))
            ,nn.LayerNorm()            
            ,Conv_trio(self.cfg,channels=1)
        ])(image)

        conved=collapse_dense(conved)


        

        return conved



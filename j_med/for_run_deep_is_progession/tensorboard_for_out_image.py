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
import tensorflow as tf
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  
import h5py
import jax
import tensorflow as tf

from jax_smi import initialise_tracking

from ..testUtils.tensorboard_utils import *

def setup_tensorboard():
    jax.numpy.set_printoptions(linewidth=400)
    ##### tensor board
    #just removing to reduce memory usage of tensorboard logs
    shutil.rmtree('/workspaces/pilot_lymphoma/data/tensor_board')
    os.makedirs("/workspaces/pilot_lymphoma/data/tensor_board")

    # profiler_dir='/workspaces/Jax_cuda_med/data/profiler_data'
    # shutil.rmtree(profiler_dir)
    # os.makedirs(profiler_dir)

    # initialise_tracking()

    logdir="/workspaces/pilot_lymphoma/data/tensor_board"
    # plt.rcParams["savefig.bbox"] = 'tight'
    file_writer = tf.summary.create_file_writer(logdir)
    return file_writer

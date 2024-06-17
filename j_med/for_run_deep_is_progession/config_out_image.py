import ml_collections
import jax
import numpy as np
from ml_collections import config_dict

def get_cfg():
    cfg = config_dict.ConfigDict()
    # cfg.total_steps=8000
    cfg.total_steps=100

    cfg.learning_rate=0.00001
    cfg.convolution_channels=20
    cfg.num_classes=2
    cfg.batch_size=2

    cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
    cfg.img_size = (cfg.batch_size,488, 128, 128,4)
    # cfg.img_size = (cfg.batch_size,488, 200, 200,2)
    # cfg.img_size = (cfg.batch_size,488, 124, 124,4)
   
    #just for numerical stability
    cfg.epsilon=0.0000000000001
    cfg.optax_name = 'big_vision.scale_by_adafactor'
    cfg.optax = dict(beta2_cap=0.95)

    cfg.lr = cfg.learning_rate
    cfg.wd = 0.00001 # default is 0.0001; paper used 0.3, effective wd=0.3*lr
    cfg.schedule = dict(
        warmup_steps=20,
        decay_type='linear',
        linear_end=cfg.lr/100,
    )

    # GSAM settings.
    # Note: when rho_max=rho_min and alpha=0, GSAM reduces to SAM.
    cfg.gsam = dict(
        rho_max=0.6,
        rho_min=0.1,
        alpha=0.6,
        lr_max=cfg.get_ref('lr'),
        lr_min=cfg.schedule.get_ref('linear_end') * cfg.get_ref('lr'),
    )

    #setting how frequent the checkpoints should be performed
    cfg.divisor_checkpoint=15
    cfg.divisor_logging=4
    cfg.to_save_check_point=True

    cfg.is_gsam=False
    cfg.num_iter_initialization=5
    # convSpecs=(
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }
    #     )
    
    # cfg.convSpecs= list(map(FrozenDict,convSpecs ))
    # self.convSpecs=[{'in_channels':3,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
    # ,{'in_channels':4,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
    # ,{'in_channels':4,'out_channels':8, 'kernel_size':(3,3,3),'stride':(1,1,1) }
    # ,{'in_channels':8,'out_channels':16, 'kernel_size':(3,3,3),'stride':(2,2,2) }]

    cfg = ml_collections.FrozenConfigDict(cfg)

    return cfg



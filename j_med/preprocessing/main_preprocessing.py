import SimpleITK as sitk
from subprocess import Popen
import subprocess
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import functools
from functools import partial
import sys
import os.path
from os import path as pathOs
import numpy as np
import tempfile
import shutil
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
import fileinput
import re
import subprocess
from toolz.itertoolz import groupby
from .elastixRegister import *
data_path="/workspaces/pilot_lymphoma/data/pilot_lymphoma_suvs"

elacticPath='/root/elastixBase/elastix-5.0.1-linux/bin/elastix'
transformix_path='/root/elastixBase/elastix-5.0.1-linux/bin/transformix'
reg_prop='/workspaces/pilot_lymphoma/j_med/preprocessing/registration/parameters.txt'  

preprocessed_path='/workspaces/pilot_lymphoma/data/preprocessed'



def copy_changing_type(source, dest):
    image= sitk.ReadImage(source)
    # nan_count=np.sum(np.isnan(np.array(sitk.GetArrayFromImage(image)).flatten()))
    # if(nan_count>0):
    #     raise ValueError(f"!!! nan in {source}")
    # image = sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    image=sitk.Cast(image, sitk.sitkFloat32)
    writer = sitk.ImageFileWriter() 
    writer.SetFileName(dest)
    writer.Execute(image)
    return dest

def register_studies(group,temp_dir,reg_prop,elacticPath,transformix_path,preprocessed_path,data_path):
    """ 
    using elastix doing very basic registration of previous study to current study
    """
    patId=group[0]
    loc_temp_dir=f"{temp_dir}/{patId}"
    paths_pair=group[1]
    paths_pair= list(map(lambda p: join(data_path,p),paths_pair))

    # regg=reg_a_to_b(loc_temp_dir,patId,paths_pair[0],paths_pair[1],[],reg_prop ,elacticPath,transformix_path,'PET',reIndex=0)

    new_path_0=join(preprocessed_path,paths_pair[0].split('/')[-1])
    new_path_1=join(preprocessed_path,paths_pair[1].split('/')[-1])
    # copy_changing_type(regg[1],new_path_0)
    copy_changing_type(paths_pair[0],new_path_0)
    copy_changing_type(paths_pair[1],new_path_1)

    return new_path_0,new_path_1



shutil.rmtree(preprocessed_path, ignore_errors=True) 
os.makedirs(preprocessed_path ,exist_ok = True)

temp_dir = tempfile.mkdtemp()
pathss=os.listdir(data_path)
grouped_by_pat_num= groupby(lambda pathh : pathh.split('_')[1] ,pathss)
# grouped_by_master=[(key,list(group)) for key, group in grouped_by_master]
grouped_by_pat_num= dict(grouped_by_pat_num).items()

preprocessed=[]
with mp.Pool(processes = mp.cpu_count()) as pool: 
    # register_studies(group,temp_dir,reg_prop,elacticPath,transformix_path,preprocessed_path,data_path)
    preprocessed=pool.map(partial(register_studies,temp_dir=temp_dir,reg_prop=reg_prop,elacticPath=elacticPath,transformix_path=transformix_path,preprocessed_path=preprocessed_path,data_path=data_path),grouped_by_pat_num )

shutil.rmtree(temp_dir, ignore_errors=True) 

print(grouped_by_pat_num)

#python3 -m j_med.preprocessing.main_preprocessing
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor, getTestCase
import itertools
import h5py
import shutil
import pandas as pd
import tempfile
import os
from functools import partial
import multiprocessing as mp


def get_extractor():
    """
    get pyradiomics extractor object
    """
    params="/workspaces/konwersjaJsonData/radiomics/Params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.enableAllFeatures()
    return extractor


def get_lesions_from_mask(mask):
    """
    return boolean masks for each lesion (so list of masks the length equal to number of inferred lesions)
    """
    connected=sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(int))))
    uniqq=np.unique(connected)
    uniqq= list(filter(lambda el:el>0,uniqq))
    return list(map(lambda i: connected==i ,uniqq))

def save_image_to_temp(image,temp_dir,name,spacing):
    """
    save image to temp file and return path
    """
    image.SetSpacing(spacing)
    temp_path=os.path.join(temp_dir,f"{name}.nii.gz")
    sitk.WriteImage(image,temp_path)
    return temp_path

def extract_for_lesion(modality,bool_lesion,extractor,spacing,mod_name,lesion_num,pat_id,min_voxels,Deauville,im_dir,study_0_or_1):
    """
    given a single modality and single lesion from h5 file, extract features for each lesion
    """ 
    
    try:
        bool_lesion_sum = np.sum(bool_lesion.flatten())
        if(bool_lesion_sum<min_voxels):
            return {}

        sitk_mod=sitk.GetImageFromArray(modality)
        sitk_les=sitk.GetImageFromArray(bool_lesion.astype(np.uint8))

        
        loc_dir=f"{im_dir}/{pat_id}/{study_0_or_1}/{mod_name}/{lesion_num}"
        os.makedirs(loc_dir, exist_ok=True)
        lab_path=save_image_to_temp(sitk_les,loc_dir,"lab",spacing)
        
        
        temp_dir=tempfile.mkdtemp()
        im_path=save_image_to_temp(sitk_mod,temp_dir,"modal",spacing)
        shutil.rmtree(temp_dir)
        
        
        extracted=extractor.execute(im_path, lab_path, 1)
        res={}
        loc_add_name=f"_{mod_name}"
        keys=list(extracted.keys())

        res[f"pat_id"]=pat_id
        res[f"lesion_num"]=lesion_num
        res[f"study_0_or_1"]=study_0_or_1
        res[f"Deauville"]=Deauville
        res[f"lab_path"]=lab_path
        res[f"mod_name"]=mod_name
        res[f"vol_in_mm3"]=np.sum(bool_lesion)*np.prod(spacing)
        

        for keyy in keys:
            if("diagnostics" not in keyy):
                res[keyy+loc_add_name]=extracted[keyy]
        # res=pd.Series(res)
    except Exception as e:
        print(f"error {e}")
        res={}
 

    return res
    


def extract_for_lesions(lesion_with_num,pet,ct,extractor,spacing,pat_id,min_voxels,Deauville,im_dir,study_0_or_1):
    """
    given a modality from h5 file, extract features for each lesion
    """   

    dicts= list(map(lambda modality_with_name:extract_for_lesion(modality_with_name[1],lesion_with_num[2]
    ,extractor,spacing,modality_with_name[0],lesion_with_num[1],pat_id,min_voxels,lesion_with_num[0],Deauville,im_dir,study_0_or_1)
                    ,[("pet",pet),("ct",ct)]))
    return {**dicts[0],**dicts[1]}



def extract_for_case(group,extractor,spacing,pat_id,min_voxels,douville_df_row):
    """
    given paths, extract features for each lesion and each modality
    """


    masks=list(map(lambda k: group[k][:],masks_keys))

    

    #we want to get the list of tuples where each tuples has 3 entries (key_name, inferred lesion number, single lesion boolean mask)
    list_bool_lesions=(mask_tupl[0],  get_lesions_from_mask(mask_tupl[1]))
    summed= np.sum(np.stack(np.stack(list_bool_lesions,axis=0)),axis=0)
    list_bool_lesions=list(map(lambda el:  list(map(lambda inner_el:(el[0],inner_el[0],inner_el[1]) ,list(enumerate(el[1])) ))    ,list_bool_lesions))
    list_bool_lesions=list(itertools.chain(*list_bool_lesions))

    res=[]
    if(len(list_bool_lesions)>0):
        with mp.Pool(processes = mp.cpu_count()) as pool:
            res=pool.map(partial(extract_for_lesions,pet=pet,ct=ct,extractor=extractor,spacing=spacing,pat_id=pat_id,min_voxels=min_voxels
                                 ,Deauville=Deauville,im_dir=im_dir,study_0_or_1=study_0_or_1),list_bool_lesions)
    
        #### adding features from all lesions at once
        res.append(extract_for_lesions((1000,summed),pet=pet,ct=ct,extractor=extractor,spacing=spacing,pat_id=pat_id,min_voxels=min_voxels
                                 ,Deauville=Deauville,im_dir=im_dir,study_0_or_1=study_0_or_1))
    return res




"""
next we need to cross check lesions of the same patient with diffrent study_0_or_1 values for any overlapping lesions
we will get those overlapping lesions by analizing cartesian product of lesions paths  (taken from dataframe we got from extract_for_case) 
    from both studies and checking if they overlap enough
we will then subtract values of radiomic features for those lesions save; get sum of absolute values of those differences 
and futher analyze delta radiomics only for those that change was biggest. we can also add some additional values 
    like tmtv for both images, and some mean texture radiomics of all lesions combined  and its change etc.

alternatively we can construct simple graph where edges between lesions will indicate physical proxumity ...
"""





extractor=get_extractor()

for_df=list(map( lambda case: extract_for_case(non_iso[case],extractor,spacing,case,is_cancer_dicts,isups_dict,fold_val,min_voxels),cases))
for_df= list(filter(lambda el:len(el)>0,for_df))
for_df=list(itertools.chain(*for_df))
for_df= list(filter(lambda el:"original_shape_Maximum2DDiameterSlice_pet" in el.keys(),for_df))
for_df= list(filter(lambda el:el["original_shape_Maximum2DDiameterSlice_pet"]!="" and el["original_shape_Maximum2DDiameterSlice_pet"]!=" ",for_df))

# print(for_df[1])
#flattening
# for_df=list(itertools.chain(*for_df))
os.makedirs("/workspaces/konwersjaJsonData/explore",exist_ok=True)
csv_res_path="/workspaces/konwersjaJsonData/explore/extracted_features_c.csv"
pd.DataFrame(for_df).to_csv(csv_res_path)
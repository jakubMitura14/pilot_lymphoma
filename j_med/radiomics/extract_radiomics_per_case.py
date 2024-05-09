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
import faulthandler
import gc
faulthandler.enable()

def get_extractor():
    """
    get pyradiomics extractor object
    """
    params="/workspaces/pilot_lymphoma/data/Params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('ngtdm')
    return extractor



# @workspace given extractor initialized by code  ```     extractor.enableAllFeatures()
#     extractor.disableFeatureClassByName('shape2D')
#     extractor.disableFeatureClassByName('shape3D')``` I want to switch off 2D and 3D shape features However current code give error  ```'RadiomicsFeatureExtractor' object has no attribute 'disableFeatureClassByName'```


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
    
    # try:
    bool_lesion_sum = np.sum(bool_lesion.flatten())
    if(bool_lesion_sum<min_voxels):
        return {}

    # print(f"mmmmm modality {type(modality)} {modality.shape} bool_lesion {type(bool_lesion)} {bool_lesion.shape} {np.sum(bool_lesion)}")
    sitk_mod=sitk.GetImageFromArray(modality)
    sitk_les=sitk.GetImageFromArray(bool_lesion.astype(np.uint8))

    
    loc_dir=f"{im_dir}/{pat_id}/{study_0_or_1}/{mod_name}/{lesion_num}"
    os.makedirs(loc_dir, exist_ok=True)
    lab_path=save_image_to_temp(sitk_les,loc_dir,"lab",spacing)
    
    
    temp_dir=tempfile.mkdtemp()
    im_path=save_image_to_temp(sitk_mod,temp_dir,"modal",spacing)
    # shutil.rmtree(temp_dir)
    
    
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
    # except Exception as e:
    #     print(f"error {e}")
    #     res={}
 

    return res
    


def extract_for_lesions(lesion_with_num,pet,ct,extractor,spacing,pat_id,min_voxels,Deauville,im_dir,study_0_or_1):
    """
    given a modality from h5 file, extract features for each lesion
    """   
    print(f"  {len(lesion_with_num)}")
    # if(len(lesion_with_num)<3):
    #     return {}
    dicts= list(map(lambda modality_with_name:extract_for_lesion(modality_with_name[1],lesion_with_num[2]
    ,extractor,spacing,modality_with_name[0],lesion_with_num[1],pat_id,min_voxels,Deauville,im_dir,study_0_or_1)
                    ,[("pet",pet),("ct",ct)]))
    return {**dicts[0],**dicts[1]}



def extract_for_case(curr_row,extractor,min_voxels,im_dir):
    """
    given paths, extract features for each lesion and each modality
    1) get study for patient and extract path to ct's and SUV images based on study_0_or_1 an patient id
    2) extract information about Deauville from each study plus study_0_or_1 info
    """
    csv_res_path="/workspaces/pilot_lymphoma/data/extracted_features_pet_full_curr.csv"
    df_created=False
    no_pat_id=False
    if(os.path.exists(csv_res_path)==False):
        pd.DataFrame().to_csv(csv_res_path)
        df_created=True
    res_csv=pd.read_csv(csv_res_path)
    if("pat_id" not in list(res_csv.keys())):
        no_pat_id=True
    curr_row=curr_row[1]
    #extract necessary information from dataframe
    pat_id= curr_row["pat_id"]
    Deauville= curr_row["deauville_1"]
    study_0_or_1= curr_row["study_0_or_1"]
    
    print(f"************** pat_id {pat_id} study_0_or_1 {study_0_or_1}")

    #check if we already have extracted features for this patient
    if((not df_created) and (not no_pat_id)):
        if(len(res_csv.loc[(res_csv["pat_id"]==pat_id) & (res_csv["study_0_or_1"]==study_0_or_1)])>0):
            print(f" pat_id {pat_id} study_0_or_1 {study_0_or_1} already extracted")
            return []
        if((pat_id==26) and (study_0_or_1==0)):
            print(f" pat_id {pat_id} study_0_or_1 {study_0_or_1} killed")
            return []

    #get paths to the files from main folder
    reg_form="lin_transf"
    path_curr_folder=f"{path_folder_files}/pat_{pat_id}/{reg_form}"
    path_SUV= f"{path_curr_folder}/study_{study_0_or_1}_SUVS.nii.gz"
    path_CT= f"{path_curr_folder}/study_{study_0_or_1}_ct_soft.nii.gz"
    path_mask= f"{path_curr_folder}/study_{study_0_or_1}_tmtvNet_SEG.nii.gz"
    if(os.path.exists(path_SUV)==False or os.path.exists(path_CT)==False or os.path.exists(path_mask)==False):
        return []
    #load the images
    img_SUV=sitk.ReadImage(path_SUV)
    spacing=img_SUV.GetSpacing()

    img_CT=sitk.ReadImage(path_CT)
    img_mask=sitk.ReadImage(path_mask)
    pet=sitk.GetArrayFromImage(img_SUV)
    ct=sitk.GetArrayFromImage(img_CT)
   

    #we want to get the list of tuples where each tuples has 3 entries (key_name, inferred lesion number, single lesion boolean mask)
    list_bool_lesions= (f"mask_{pat_id}_{study_0_or_1}",  get_lesions_from_mask(sitk.GetArrayFromImage(img_mask)))
    # print(f"gggg {list(map(lambda el: el.shape ,list_bool_lesions[1]))}")
    # list_bool_lesions=list(itertools.chain(*list_bool_lesions))
    res=[]
    if(len(list_bool_lesions[1])>0):
        summed= np.sum(np.stack(np.stack(list_bool_lesions[1],axis=0)),axis=0)
        list_bool_lesions=list(map(lambda inner_el:(list_bool_lesions[0],inner_el[0],inner_el[1]) ,list(enumerate(list_bool_lesions[1])) ))
        gc.collect()


        # with mp.Pool(processes = mp.cpu_count()) as pool:
        #     res=pool.map(partial(extract_for_lesions,pet=pet,ct=ct,extractor=extractor,spacing=spacing,pat_id=pat_id,min_voxels=min_voxels
        #                          ,Deauville=Deauville,im_dir=im_dir,study_0_or_1=study_0_or_1),list_bool_lesions)
        res=list(map(partial(extract_for_lesions,pet=pet,ct=ct,extractor=extractor,spacing=spacing,pat_id=pat_id,min_voxels=min_voxels
                                ,Deauville=Deauville,im_dir=im_dir,study_0_or_1=study_0_or_1),list_bool_lesions))
           
        #### adding features from all lesions at once
        res.append(extract_for_lesions((f"mask_{pat_id}_{study_0_or_1}",1000,summed),pet=pet,ct=ct,extractor=extractor,spacing=spacing,pat_id=pat_id,min_voxels=min_voxels
                                 ,Deauville=Deauville,im_dir=im_dir,study_0_or_1=study_0_or_1))
        gc.collect()

        ############3333
        res= list(filter(lambda el:len(el)>0,res))
        # res= list(filter(lambda el:"original_shape_Maximum2DDiameterSlice_pet" in el.keys(),res))
        # res= list(filter(lambda el:el["original_shape_Maximum2DDiameterSlice_pet"]!="" and el["original_shape_Maximum2DDiameterSlice_pet"]!=" ",res))

        
        df_to_append=pd.DataFrame(res)
        for_df=df_to_append
        print(f"ddddddf_to_append {len(df_to_append)} res {len(res)}")
        if(not df_created):
            for_df=res_csv.append(df_to_append)

        # print(for_df[1])
        #flattening
        # for_df=list(itertools.chain(*for_df))
        # os.makedirs("/workspaces/konwersjaJsonData/explore",exist_ok=True)

        pd.DataFrame(for_df).to_csv(csv_res_path)



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



path_deauvill_df="/workspaces/pilot_lymphoma/data/all_deauville_form.csv"
path_folder_files="/root/data/prepared_registered"
im_dir="/workspaces/pilot_lymphoma/data/for_radiomics_folder"

deauvill_df=pd.read_csv(path_deauvill_df)

# curr_row=list(deauvill_df.iterrows())[0][1]
min_voxels=4

for_df=[]
rows=list(deauvill_df.iterrows())
for_df=list(map(partial(extract_for_case,extractor=extractor,min_voxels=min_voxels,im_dir=im_dir),rows))


# for_df=list(map( lambda case: extract_for_case(non_iso[case],extractor,spacing,case,is_cancer_dicts,isups_dict,fold_val,min_voxels),cases))
for_df= list(filter(lambda el:len(el)>0,for_df))
for_df=list(itertools.chain(*for_df))
for_df= list(filter(lambda el:"original_shape_Maximum2DDiameterSlice_pet" in el.keys(),for_df))
for_df= list(filter(lambda el:el["original_shape_Maximum2DDiameterSlice_pet"]!="" and el["original_shape_Maximum2DDiameterSlice_pet"]!=" ",for_df))

# print(for_df[1])
#flattening
# for_df=list(itertools.chain(*for_df))
# os.makedirs("/workspaces/konwersjaJsonData/explore",exist_ok=True)
csv_res_path="/workspaces/pilot_lymphoma/data/extracted_features_pet_full.csv"
pd.DataFrame(for_df).to_csv(csv_res_path)
#!/bin/python
from nilearn import image
anat_file = '/home/clancy/data/GSP1000/GSP1000_v2_dataset/sub-0001/anat/sub-0001_norm_MNI152_1mm.nii.gz'
func_file = '/home/clancy/data/GSP1000/GSP1000_v2_dataset/sub-0001/func/sub-0001_bld001_rest_skip4_stc_mc_bp_0.0001_0.08_resid_FS1mm_MNI1mm_MNI2mm_sm7_finalmask.nii.gz'
fslinfo = '/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz'
fslinfo_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'

image.load_img(anat_file).affine
image.load_img(func_file).affine
image.load_img(fslinfo).affine
image.load_img(fslinfo_mask).affine



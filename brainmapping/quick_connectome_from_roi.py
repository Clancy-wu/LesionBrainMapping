#!/bin/python
import os
import subprocess
import pandas as pd
from nilearn import image, maskers
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import itertools

os.environ['GSP1000_DATA'] = '/home/clancy/data/GSP1000/GSP1000_v2_dataset' # setup gsp dataset

def run(f, this_iter, n_threads):
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        results = list(tqdm(executor.map(f, this_iter), total=len(this_iter)))
    return results

def coord2voxel(my_coord, img_affine):
    mni_x, mni_y, mni_z = my_coord[0], my_coord[1], my_coord[2]
    coords = np.c_[
        np.atleast_1d(mni_x).flat,
        np.atleast_1d(mni_y).flat,
        np.atleast_1d(mni_z).flat,
        np.ones_like(np.atleast_1d(mni_z).flat),
    ].T
    project_affine = np.linalg.inv(img_affine)
    voxel_x, voxel_y, voxel_z, _ = np.around(np.dot(project_affine, coords))
    return voxel_x.item(), voxel_y.item(), voxel_z.item()

def fsl_mni152_2mm():
    mni_template = subprocess.check_output('echo $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz', shell=True, text=True) # 91, 109, 91
    mni_template = mni_template.replace('\n','')
    mni_img = image.load_img(mni_template)
    return mni_img

def compute_fr_fz(matrix_a, matrix_b):
    mean_a = np.mean(matrix_a, axis=0)
    mean_b = np.mean(matrix_b, axis=0)
    std_a = np.std(matrix_a, axis=0)
    std_b = np.std(matrix_b, axis=0)
    covariance_matrix = np.mean((matrix_a - mean_a) * (matrix_b - mean_b), axis=0)
    np.seterr(invalid='ignore') # ignore the warning message because 0/0 = nan.
    fr = covariance_matrix / (std_a * std_b)
    fr[np.isnan(fr)] = 0
    fz = np.arctanh(fr)
    return fz

def single_matrix_from_single_roi(roi_img_num, health_path, mni152_masker):
    health_nii_num = mni152_masker.transform(health_path)
    roi_img_mean = np.mean(roi_img_num * health_nii_num, axis=1)
    roi_img_num_inverse = 1 - roi_img_num
    roi_health_data = roi_img_num_inverse * health_nii_num
    roi_img_mean_shape = roi_img_mean[:, np.newaxis]
    roi_mean_num = np.tile(roi_img_mean_shape, roi_health_data.shape[1])
    roi_mean_data = roi_img_num_inverse * roi_mean_num
    roi_single_fz = compute_fr_fz(roi_health_data, roi_mean_data)
    return roi_single_fz

def single_matrix_from_single_roi_batch(args):
    return single_matrix_from_single_roi(*args)

def quick_connectome_from_roi(lesion_dir, output_dir, n_threads):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    lesion_files = [os.path.join(lesion_dir, x)  for x in os.listdir(lesion_dir) if 'nii' in x]
    if len(lesion_files) < 1:
        raise Exception('No ROI found in the directory.')
    gsp_dir = os.environ['GSP1000_DATA']
    health_nii_files = glob(os.path.join(gsp_dir, '*', 'func', '*bld001*.nii.gz'))
    mni152_2mm = fsl_mni152_2mm()
    for lesion_file in lesion_files:
        mni152_masker = maskers.NiftiMasker(mask_img=mni152_2mm, smoothing_fwhm=None, standardize=False, 
                                    standardize_confounds=False, detrend=False,
                                    low_pass=None, high_pass=None, t_r=None)
        roi_img_num = mni152_masker.fit_transform(lesion_file)
        compute_lesion_iters = list(itertools.product([roi_img_num], health_nii_files, [mni152_masker]))
        lesion_fz_healths = run(single_matrix_from_single_roi_batch, compute_lesion_iters, n_threads)
        lesion_fz_healths_matrix = np.vstack(lesion_fz_healths)
        avgR_fz_num = np.mean(lesion_fz_healths_matrix, axis=0)
        avgR_num = np.mean(np.tanh(lesion_fz_healths_matrix), axis=0)
        sample_std = np.std(lesion_fz_healths_matrix, ddof=1, axis=0)
        np.seterr(invalid='ignore') # ignore the warning message because 0/0 = nan.
        std_error = sample_std / np.sqrt(lesion_fz_healths_matrix.shape[0])
        T_num = avgR_fz_num / std_error
        T_num[np.isnan(T_num)] = 0        
        lesion_name = os.path.basename(lesion_file).split('.nii')[0]
        mni152_masker.inverse_transform(avgR_fz_num).to_filename(os.path.join(output_dir, lesion_name+'_AvgR_Fz.nii.gz'))
        mni152_masker.inverse_transform(avgR_num).to_filename(os.path.join(output_dir, lesion_name+'_AvgR.nii.gz'))
        mni152_masker.inverse_transform(T_num).to_filename(os.path.join(output_dir, lesion_name+'_T.nii.gz'))
        print(f'ROI: {lesion_name} has been successfully computed.')

def quick_connectome_from_coordinate(lesion_MNI_coordinate_csv, roi_radius, output_dir, n_threads):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    csv_data = pd.read_csv(lesion_MNI_coordinate_csv)
    duplicated_label = csv_data['label'].duplicated().sum()
    if duplicated_label > 0:
        raise Exception('Your csv has same label, please change the label to be unique.')
    gsp_dir = os.environ['GSP1000_DATA']
    #health_nii_files = glob(os.path.join(gsp_dir, '*', 'func', '*bld001*.nii.gz'))
    health_nii_files = glob(os.path.join(gsp_dir, '*', 'func', '*bld001*.nii.gz'))
    mni152_2mm = fsl_mni152_2mm()
    for i in range(csv_data.shape[0]):  
        coord_name = str(csv_data['label'][i])
        coord_x = csv_data['x'][i]
        coord_y = csv_data['y'][i]
        coord_z = csv_data['z'][i]
        voxcoords = coord2voxel([coord_x, coord_y, coord_z], mni152_2mm.affine)
        lesion_file = os.path.join(output_dir, coord_name+'.nii.gz')
        command_1 = "fslmaths $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask_dil -mul 0 -add 1 -roi %s 1 %s 1 %s 1 0 1 %s -odt float" %(voxcoords[0],voxcoords[1],voxcoords[2],lesion_file)
        command_2 = "fslmaths %s -kernel sphere %s -fmean %s -odt float" % (lesion_file, roi_radius, lesion_file)
        command_3 = "fslmaths %s -bin %s" % (lesion_file, lesion_file)
        os.system(command_1)
        os.system(command_2)
        os.system(command_3)
        mni152_masker = maskers.NiftiMasker(mask_img=mni152_2mm, smoothing_fwhm=None, standardize=False, 
                                    standardize_confounds=False, detrend=False,
                                    low_pass=None, high_pass=None, t_r=None)
        roi_img_num = mni152_masker.fit_transform(lesion_file)
        compute_lesion_iters = list(itertools.product([roi_img_num], health_nii_files, [mni152_masker]))
        lesion_fz_healths = run(single_matrix_from_single_roi_batch, compute_lesion_iters, n_threads)
        lesion_fz_healths_matrix = np.vstack(lesion_fz_healths)
        avgR_fz_num = np.mean(lesion_fz_healths_matrix, axis=0)
        avgR_num = np.mean(np.tanh(lesion_fz_healths_matrix), axis=0)
        sample_std = np.std(lesion_fz_healths_matrix, ddof=1, axis=0)
        np.seterr(invalid='ignore') # ignore the warning message because 0/0 = nan.
        std_error = sample_std / np.sqrt(lesion_fz_healths_matrix.shape[0])
        T_num = avgR_fz_num / std_error
        T_num[np.isnan(T_num)] = 0
        lesion_name = os.path.basename(lesion_file).split('.nii')[0]
        mni152_masker.inverse_transform(avgR_fz_num).to_filename(os.path.join(output_dir, lesion_name+'_AvgR_Fz.nii.gz'))
        mni152_masker.inverse_transform(avgR_num).to_filename(os.path.join(output_dir, lesion_name+'_AvgR.nii.gz'))
        mni152_masker.inverse_transform(T_num).to_filename(os.path.join(output_dir, lesion_name+'_T.nii.gz'))
        print(f'ROI: {lesion_name} has been successfully computed.')

# author@kangwu
# date: Feb 27 24        

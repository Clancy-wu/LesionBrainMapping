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
import re
import warnings

os.environ['GSP1000_DATA'] = '/home/clancy/data/GSP1000/GSP1000_v2_dataset' # setup gsp dataset

def run(f, this_iter, n_threads):
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        results = list(tqdm(executor.map(f, this_iter), total=len(this_iter)))
    return results

def mmToVox_fsleyes_2mm(mmcoords):
    voxcoords = ['','','']
    voxcoords[0] = str(int(round(mmcoords[0]/2))*-1+45)
    voxcoords[1] = str(int(round(mmcoords[1]/2))+63)
    voxcoords[2] = str(int(round(mmcoords[2]/2))+36)
    return voxcoords

def fsl_mni152_2mm():
    mni_template = subprocess.check_output('echo $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz', shell=True, text=True) # 91, 109, 91
    mni_template = mni_template.replace('\n','')
    mni_img = image.load_img(mni_template)
    return mni_img

def make_sphere_from_coords_in_csv(csv_file, roi_radius, output_dir):
    csv_data = pd.read_csv(csv_file)
    for i in range(csv_data.shape[0]):  
        coord_name = str(csv_data['Label'][i])
        coord_x = csv_data['X'][i]
        coord_y = csv_data['Y'][i]
        coord_z = csv_data['Z'][i]
        voxcoords = mmToVox_fsleyes_2mm([coord_x, coord_y, coord_z])
        command_1 = "fslmaths $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask -mul 0 -add 1 -roi %s 1 %s 1 %s 1 0 1 tmp -odt float" %(voxcoords[0],voxcoords[1],voxcoords[2])
        command_2 = "fslmaths tmp -kernel sphere %s -fmean tmp -odt float" % (roi_radius)
        command_3 = "fslmaths tmp -bin %s" % os.path.join(output_dir, coord_name+'.nii.gz')
        command_4 = "rm tmp.nii.gz"
        os.system(command_1)
        os.system(command_2)
        os.system(command_3)
        os.system(command_4)

def make_sphere_from_maps(maps_dir, threshold, output_dir):
    maps_nii = os.listdir(maps_dir)
    for i in range(len(maps_nii)):
        org_img = os.path.join(maps_dir, maps_nii[i])
        trg_img = os.path.join(output_dir, 'roi_'+maps_nii[i])
        command_3 = "fslmaths %s -thr %s -bin %s" % (org_img, threshold, trg_img)
        os.system(command_3)

def maps_intensity_from_coord(mni_coord_csv, roi_radius, maps_dir):
    mni152_img = fsl_mni152_2mm()
    maps_all_data = np.zeros_like(mni152_img.get_fdata())
    for map in os.listdir(maps_dir):
        map_nii = image.load_img(os.path.join(maps_dir, map))
        if map_nii.affine != mni152_img.affine:
            stand_map_nii = image.resample_img(map_nii, target_affine=mni152_img.affine, target_shape=mni152_img.shape, interpolation='nearest')
        else:
            stand_map_nii = map_nii
        maps_all_data += stand_map_nii.get_fdata()
    maps_all = image.new_img_like(mni152_img, data=maps_all_data, affine=mni152_img.affine)
    csv_data = pd.read_csv(mni_coord_csv)
    csv_intensity = []; csv_percentage = []
    for i in range(csv_data.shape[0]):
        coord_x = float(csv_data['X'][i])
        coord_y = float(csv_data['Y'][i])
        coord_z = float(csv_data['Z'][i])
        coord_seed = [(coord_x, coord_y, coord_z)]
        masker_seed = maskers.NiftiSpheresMasker(seeds=coord_seed, radius=roi_radius, standardize_confounds=False)
        data_seed_mean = masker_seed.fit_transform(maps_all)
        csv_intensity.append(round(data_seed_mean[0,0], 4))
        data_seed_percent = round(data_seed_mean[0,0] / np.max(maps_all_data) * 100, 2)
        csv_percentage.append(data_seed_percent)
    csv_data['intensity'] = csv_intensity; csv_data['percentage'] = csv_percentage
    csv_data.to_csv('coordinate_explore_result.csv', header=True, index=None)

def single_func_zmap_from_single_roi(roi_nii, health_nii, roi_output_dir, mni152_img):
    roi_img = image.load_img(roi_nii)
    mni152_masker = maskers.NiftiMasker(mask_img=mni152_img, smoothing_fwhm=None, standardize=False, 
                                standardize_confounds=False, detrend=False,
                                low_pass=None, high_pass=None, t_r=None )
    roi_mask = mni152_masker.fit_transform(roi_img)
    func_data = mni152_masker.fit_transform(health_nii)
    roi_data_mean = np.mean(roi_mask * func_data, axis=1)
    roi_mask_inverse = 1 - roi_mask
    roi_data_inverse = roi_mask_inverse * func_data
    data_R = np.zeros_like(roi_mask)
    np.seterr(invalid='ignore') # ignore the warning message because 0/0 = nan in the next command.
    for i in range(roi_data_inverse.shape[1]):
        data_R[0,i] = round(np.corrcoef(roi_data_inverse[:,i], roi_data_mean)[0,1], 6)
    data_R = np.nan_to_num(data_R, nan=0.0)
    data_Z = np.arctanh(data_R)
    nii_Z = mni152_masker.inverse_transform(data_Z)
    nii_Z_name = os.path.join(roi_output_dir, 'fz_'+os.path.basename(health_nii))
    nii_Z.to_filename(nii_Z_name)

def single_func_zmap_from_single_roi_batch(args):
    return single_func_zmap_from_single_roi(*args)

def gsp_health_zmap_from_rois(roi_dir, gsp_dir, output_dir, n_threads):
    roi_niis = glob(os.path.join(roi_dir, '*nii*'))
    health_niis = glob(os.path.join(gsp_dir, 'sub-*', 'func', '*bld001_*.nii.gz'))
    mni152_img = fsl_mni152_2mm()
    for roi_nii in roi_niis:
        roi_basename = os.path.basename(roi_nii).split('.nii')[0]
        roi_output_dir = os.path.join(output_dir, roi_basename)
        if not os.path.exists(roi_output_dir):
            os.mkdir(roi_output_dir)
        roi_healths_batch = list(itertools.product([roi_nii], health_niis, [roi_output_dir], [mni152_img]))
        run(single_func_zmap_from_single_roi_batch, roi_healths_batch, n_threads)
        fz_maps = glob(os.path.join(roi_output_dir, 'fz_*'))
        fz_maps.sort(key=lambda x: int("".join(re.findall("\d+",x))))
        fz_4d = image.concat_imgs(fz_maps)
        fz_4d_name = os.path.join(roi_output_dir, 'fz_map_all.nii')
        fz_4d.to_filename(fz_4d_name)
        [os.remove(x) for x in fz_maps]

def calculate_network_damage_degree(test_lesion, lesion_network, type='both'):
    test_lesion_img = image.load_img(test_lesion)
    lesion_network_img = image.load_img(lesion_network)
    if test_lesion_img.affine != lesion_network_img.affine or test_lesion_img.shape != lesion_network_img.shape:
        test_lesion_img_resample = image.resample_to_img(test_lesion_img, lesion_network_img, interpolation='nearest')    
    else:
        test_lesion_img_resample = test_lesion_img
    test_lesion_data = test_lesion_img_resample.get_fdata()
    lesion_network_data = lesion_network_img.get_fdata()
    masked_data = np.multiply(test_lesion_data, lesion_network_data)
    if test_lesion_data.min() < 0:
        warnings.warn('WARNING: negative weights are detected, be care of the input type', UserWarning)
    if type == 'positive':
        threshold_mask = np.where(masked_data>0, masked_data, np.nan)
    if type == 'negative':
        threshold_mask = np.where(masked_data<0, masked_data, np.nan)
    else:
        threshold_mask = masked_data
    damage_intensity = round(np.nansum((threshold_mask)), 4)
    damage_percent = round(100 - (np.sum(np.isnan(threshold_mask)) / np.size(threshold_mask) * 100), 2)
    return damage_intensity, damage_percent

def compute_onesample_randomize_map_single(roi_fz_map_file, iterations):
    roi_fz_map_dir = os.path.dirname(roi_fz_map_file)
    roi_fz_average_map = image.mean_img(roi_fz_map_file)
    roi_fz_average_map_name = os.path.join(roi_fz_map_dir, 'average_fz_map_all.nii.gz')
    roi_fz_average_map.to_filename(roi_fz_average_map_name)
    if 'nii.gz' in roi_fz_map_file:
        roi_fz_map_file_nii = roi_fz_map_file.replace('.gz', '')
        roi_fz_map_file_nii_img = image.load_img(roi_fz_map_file)
        roi_fz_map_file_nii_img.to_filename(roi_fz_map_file_nii)
    else:
        roi_fz_map_file_nii = roi_fz_map_file
    randomise_output = os.path.join(roi_fz_map_dir, 'onesample_fzmap')
    randomise_command = f"randomise -i {roi_fz_map_file_nii} -o {randomise_output} -1 -T -n {str(iterations)}"
    os.system(randomise_command)

def compute_onesample_randomize_map_batch(args):
    compute_onesample_randomize_map_single(*args)

def compute_onesample_randomize_map(roi_fz_map_files, n_threads, iterations=1000):
    roi_fz_map_batch = list(itertools.product(roi_fz_map_files, [str(iterations)]))
    run(compute_onesample_randomize_map_batch, roi_fz_map_batch, n_threads)

def sum_imgs(tmap_imgs):
    mni152_img = fsl_mni152_2mm()
    sum_img = np.zeros(mni152_img.shape)
    for tmap_img in tmap_imgs:
        sum_img += tmap_img.get_fdata()
    sum_tmap = image.new_img_like(mni152_img, sum_img)
    return sum_tmap

def overlap_maps(imgs, t_level):
    pos_imgs = []
    neg_imgs = []
    for tmap in imgs:
        pos_thres_img = image.math_img('img > '+str(t_level), img = tmap)
        neg_thres_img = image.math_img('img < -'+str(t_level), img = tmap)
        pos_imgs.append(pos_thres_img)
        neg_imgs.append(neg_thres_img)
    return sum_imgs(pos_imgs), sum_imgs(neg_imgs)

def create_overlap_maps(tmap_files, output_dir, t_level=0.95):
    overlap_img = overlap_maps(tmap_files, t_level)
    overlap_img_pos_name = os.path.join(output_dir, 'pos_overlap_T-'+str(t_level)+'.nii.gz')
    overlap_img_neg_name = os.path.join(output_dir, 'neg_overlap_T-'+str(t_level)+'.nii.gz')
    overlap_img[0].to_filename(overlap_img_pos_name)
    overlap_img[1].to_filename(overlap_img_neg_name)
    percent_pos = image.math_img('np.divide(img1, '+str(len(tmap_files))+')', img1=overlap_img[0])
    percent_pos_name = os.path.join(output_dir, 'percent_pos_overlap_T-'+str(t_level)+'.nii.gz')
    percent_pos.to_filename(percent_pos_name)
    percent_neg = image.math_img('np.divide(img1, '+str(len(tmap_files))+')', img1=overlap_img[1])
    percent_neg_name = os.path.join(output_dir, 'percent_neg_overlap_T-'+str(t_level)+'.nii.gz')
    percent_neg.to_filename(percent_neg_name)
    
def generate_design_matrix(output_dir, input_files, lesion_map, add_lesion_weights=True):
    design_df = pd.DataFrame()
    input_files_name = glob(os.path.join(input_files, '*.nii*'))
    input_files_name_base = [os.path.basename(x) for x in input_files_name]
    design_df['input_files'] = input_files_name_base
    design_df['group'] = 'NA'
    design_df.to_csv(os.path.join(output_dir, 'design_matrix.csv'), index=None)
    lesion_map_img = image.load_img(lesion_map)
    lesion_map_img_binary = image.binarize_img(lesion_map_img, threshold=0)
    lesion_masker = maskers.NiftiMasker(mask_img=lesion_map_img_binary, smoothing_fwhm=None, standardize=False, 
                            standardize_confounds=False, detrend=False, high_variance_confounds=False, 
                            low_pass=None, high_pass=None, t_r=None, target_affine=lesion_map_img.affine, 
                            target_shape=None, dtype=None, reports=False)
    lesion_weights = lesion_masker.fit_transform(lesion_map_img)
    all_imgs = []
    if add_lesion_weights:
        for input_file in input_files_name:
            input_file_img = image.load_img(input_file)
            input_file_img_mean = image.mean_img(input_file_img)
            input_file_img_lesion = lesion_masker.fit_transform(input_file_img_mean)
            input_file_img_lesion_weights = np.multiply(input_file_img_lesion, lesion_weights)
            input_file_img_lesion_weights_img = lesion_masker.inverse_transform(input_file_img_lesion_weights)
            all_imgs.append(input_file_img_lesion_weights_img)
    else:
        for input_file in input_files_name:
            input_file_img = image.load_img(input_file)
            input_file_img_mean = image.mean_img(input_file_img)
            input_file_img_lesion = lesion_masker.fit_transform(input_file_img_mean)
            input_file_img_lesion_noweights_img = lesion_masker.inverse_transform(input_file_img_lesion)
            all_imgs.append(input_file_img_lesion_noweights_img)
    all_imgs_4d = image.concat_imgs(all_imgs)
    all_imgs_4d_name = os.path.join(output_dir, 'input_files_data_weight_'+str(add_lesion_weights)+'.nii')
    all_imgs_4d.to_filename(all_imgs_4d_name)
    print('Check your design matrix csv in output directory.\nChange values at group column of the csv. \
          \nAdd other covariance columns after group if you want.')

def call_palm(input_imgs, design_matrix, contrast_matrix, out_dir, iterations,
                save_1p, logp, tfce, two_tailed_flag, corrcon_flag, fdr_flag):
    palm_cmd = [
        "palm",
        "-i", input_imgs,
        "-o", out_dir+'/twosample',
        "-d", design_matrix,
        "-t", contrast_matrix,
        "-n", str(iterations)
    ]
    # Optional arguments
    if(save_1p):
        palm_cmd += ['-save1-p']
    if(logp):
        palm_cmd += ['-logp']
    if(tfce):
        palm_cmd += ['-T']
    if(two_tailed_flag):
        palm_cmd += ['-twotail']
    if(corrcon_flag):
        palm_cmd += ['-corrcon']
    if(fdr_flag):
        palm_cmd += ['-fdr']
    final_palm_cmd = ' '.join(palm_cmd)
    print("PALM with following command: ")
    print(final_palm_cmd)
    os.system(final_palm_cmd)

def run_palm(palm_analysis_dir, contrast_matrix_file, iterations=1000, save_1p=True, logp=False,
             tfce=False, two_tailed_flag=True, corrcon_flag=False, fdr_flag=False):
    data = glob(os.path.join(palm_analysis_dir, 'input_files_data*.nii'))
    design = glob(os.path.join(palm_analysis_dir, 'design_matrix.csv'))
    if len(data) != 1:
        raise Exception('No correct input data for analysis found.\nIt should be one file with name of input_files_data*.nii')
    if len(design) != 1:
        raise Exception('No correct design matrix for analysis found.\nIt should be one file with name of design_matrix.csv')
    design_data = pd.read_csv(design[0])
    design_groups = np.unique(design_data['group'])
    design_matrix_g1 = np.where(design_data['group']==design_groups[0], 1, 0)
    design_matrix_g2 = np.where(design_data['group']==design_groups[1], 1, 0)
    design_matrix = np.column_stack((design_matrix_g1, design_matrix_g2))
    if design_data.shape[1] > 2:
        design_matrix_confound = design_data.iloc[:, 2:design_data.shape[1]]
        design_matrix_confound = np.asarray(design_matrix_confound)
        design_matrix_confound = np.column_stack((design_matrix, design_matrix_confound))
    else:
        design_matrix_confound = design_matrix
    design_output = pd.DataFrame(design_matrix_confound)
    design_output_name = os.path.join(palm_analysis_dir, 'design.txt')
    design_output.to_csv(design_output_name, header=None, index=None, sep='\t')
    design_output_name_mat = design_output_name.replace('txt', 'mat')
    os.system(f"Text2Vest {design_output_name} {design_output_name_mat}")
    os.remove(design_output_name)
    contrast_matrix_con = os.path.join(palm_analysis_dir, 'contrast.con')
    os.system(f"Text2Vest {contrast_matrix_file} {contrast_matrix_con}")
    call_palm(data[0], design_output_name_mat, contrast_matrix_con, palm_analysis_dir, iterations,
                save_1p, logp, tfce, two_tailed_flag, corrcon_flag, fdr_flag)

def check_size_and_binary(lesion_mask_dir, output_dir):
    lesion_masks = glob(os.path.join(lesion_mask_dir, '*nii*'))
    temp_dir = os.path.join(output_dir, 'temp')
    os.mkdir(temp_dir)
    for lesion_mask in lesion_masks:
        nii_img = image.load_img(lesion_mask)
        mni152 = fsl_mni152_2mm()
        if len(nii_img.shape) != 3:
            raise Exception('the input lesion file is not 3D image, please check.')
        if np.unique(nii_img.get_fdata())[0] != 0 or np.unique(nii_img.get_fdata())[1] != 1:
            raise Exception('the input lesion file is not a mask, please check.\nYou can use 1_mask_priori_ROI.py to binary your lesion file.')
        else:
            nii_img_resample = image.resample_to_img(nii_img, mni152, interpolation='nearest')
            nii_img_resample_name = os.path.join(temp_dir, os.path.basename(lesion_mask))
            nii_img_resample.to_filename(nii_img_resample_name)
    return temp_dir
####################

####################
 

















    
    gsp_health_zmap_from_rois(roi_temp_dir, gsp_dir, output_dir, n_threads)
    os.system(f"rm -r {roi_temp_dir}")
    roi_fz_map_files = glob(os.path.join(output_dir, '*', 'fz_map_all.nii'))
    compute_onesample_randomize_map(roi_fz_map_files, n_threads)
    print('successfully finished.')
    
# end. author@kangwu

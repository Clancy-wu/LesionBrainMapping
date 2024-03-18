#!/bin/python
## This script is for preprocessing data from original file to output.
## Steps of the script include: if the list of lesions/MNI coords are given,
## 1. create lesion mask 2. randomise analysis 3. create tmap for each lesion

#################################################
##  Definition Area
#################################################
lesion_dir = '/home/clancy/data/LesionBrainMapping/ROIs'
output_dir = '/home/clancy/data/LesionBrainMapping/JanTest'
n_threads = 10     # n_threads number, correspondence to number of lesions
#################################################
##  Run
#################################################
from brainmapping.quick_connectome_from_roi import *
quick_connectome_from_roi(lesion_dir, output_dir, n_threads)




#################################################
##  Definition Area
#################################################
# lesion_MNI_coordinate_csv format:
# example:
# label  x  y  z
# xx, 8.67, 69.95, -2.67
# ..,  .. ,    ..,    .. 
lesion_MNI_coordinate_csv = '/home/clancy/data/LesionBrainMapping/mni_coordinate.csv'
roi_radius = 6
output_dir = '/home/clancy/data/LesionBrainMapping/JanTest'
n_threads = 10     # n_threads number, correspondence to number of lesions
#################################################
##  Run
#################################################
from brainmapping.quick_connectome_from_roi import *
quick_connectome_from_coordinate(lesion_MNI_coordinate_csv, roi_radius, output_dir, n_threads)

# author@kangwu
# date: Feb 27 24
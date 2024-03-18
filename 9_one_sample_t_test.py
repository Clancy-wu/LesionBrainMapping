#!/bin/python
## One Sample T-Test for Lesion Network Analysis
#################################################
##  Definition Area
#################################################
roi_fz_map_files = '/home/clancy/data/LesionBrainMapping/output_test/*/fz_map_all.nii.gz'
n_threads = 10     # n_threads number
#################################################
##  Run
#################################################
from glob import glob
import os
roi_fz_map_files = glob(roi_fz_map_files)
from brainmapping.functions import compute_onesample_randomize_map
compute_onesample_randomize_map(roi_fz_map_files, n_threads, iterations=1000) # 1000 is enough
## end. author@kangwu
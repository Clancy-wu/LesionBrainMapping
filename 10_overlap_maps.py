#!/bin/python
## One Sample T-Test for Lesion Network Analysis
#################################################
##  Definition Area
#################################################
roi_tmap_path = '/home/clancy/data/LesionBrainMapping/output_test'
output_dir = '/home/clancy/data/LesionBrainMapping/output_test'
t_level = 0.95 # t_level = 0.95, 0.99, 0.999; P<0.05, P<0.01, p<0.001
#################################################
##  Run
#################################################
from glob import glob
import os
tmap_files = glob(os.path.join(roi_tmap_path, '*', '*tfce_corrp_tstat1*'))
from brainmapping.functions import create_overlap_maps
create_overlap_maps(tmap_files, output_dir, t_level=0.95)
## end. author@kangwu
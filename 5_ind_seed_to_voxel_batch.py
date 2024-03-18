#!/bin/python
# individualized Seed-to-Voxel Calculations
#################################################
##  Definition Area
#################################################
roi_dir = '/home/clancy/data/LesionBrainMapping/ROIs'
gsp_dir = '/home/clancy/data/GSP1000/GSP1000_v2_dataset'
output_dir = '/home/clancy/data/LesionBrainMapping/output_test'
n_threads = 10     # n_threads number
#################################################
##  Run
#################################################
import os
from brainmapping.functions import gsp_health_zmap_from_rois
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
gsp_health_zmap_from_rois(roi_dir, gsp_dir, output_dir, n_threads)
# Baseline Test:
#     a total of 10 cores costs 1 hour per ROI.
## end. author@kangwu
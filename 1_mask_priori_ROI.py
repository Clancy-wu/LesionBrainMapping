#!/bin/python
#################################################
##  1. create ROI mask from coordinates
##  Definition Area
#################################################
output_dir = '/home/clancy/data/LesionBrainMapping/ROIs'
csv_file = '/home/clancy/data/LesionBrainMapping/sample_data/sample_coordinates.csv'
roi_radius = 6
#################################################
##  Run
#################################################
import os
from brainmapping.functions import make_sphere_from_coords_in_csv
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
make_sphere_from_coords_in_csv(csv_file, roi_radius, output_dir)

#################################################
##  2. create ROI mask from maps
##  Definition Area
#################################################
maps_dir = '/home/clancy/data/LesionBrainMapping/ROIs'
threshold = 0.5
output_dir = '/home/clancy/data/LesionBrainMapping/ROIs'
#################################################
##  Run
#################################################
import os
from brainmapping.functions import make_sphere_from_maps
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
make_sphere_from_maps(maps_dir, threshold, output_dir)

## end. author@kangwu
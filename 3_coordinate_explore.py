#!/bin/python
#################################################
##  1. Coordinate explore
##  Definition Area
##  Description: compute the intensity and percentage
##  in a series of ROIs for a given coordinate.
#################################################
mni_coord_csv = '/home/clancy/data/LesionBrainMapping/sample_data/sample_coordinates.csv'
roi_radius = 6
maps_dir = '/home/clancy/data/LesionBrainMapping/ROIs'
#################################################
##  Run
#################################################
from brainmapping.functions import maps_intensity_from_coord
maps_intensity_from_coord(mni_coord_csv, roi_radius, maps_dir)

## end. author@kangwu
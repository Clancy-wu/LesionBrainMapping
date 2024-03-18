#!/bin/python
## PALM analysis
## Reference: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM
#################################################
# Step 1: generate design csv to fill with covariances.
#################################################
##  Definition Area
#################################################
output_dir = '/home/clancy/data/LesionBrainMapping/output_test'
lesion_map = '/home/clancy/data/LesionBrainMapping/output_test/sum_overlap_T.nii.gz'
input_files = '/home/clancy/data/LesionBrainMapping/test_data/Functional_Connectivity'
#################################################
##  Run
#################################################
import os
palm_analysis = os.path.join(output_dir, 'palm_analysis')
if not os.path.exists(palm_analysis):
    os.makedirs(palm_analysis)
else:
    os.mkdir(palm_analysis)
from brainmapping.functions import generate_design_matrix
generate_design_matrix(palm_analysis, input_files, lesion_map, add_lesion_weights=True)
# end. author@kangwu
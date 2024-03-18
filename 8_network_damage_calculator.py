#!/bin/python
## Network Damage Calculation
## To get the degree of network damage status and do dose-response relationship

#################################################
##  Definition Area
#################################################
test_lesion = '/home/clancy/data/LesionBrainMapping/ROIs/ROI-1.nii.gz'
lesion_network = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
#################################################
##  Run
#################################################
from brainmapping.functions import calculate_network_damage_degree
damage_intensity, damage_percent = calculate_network_damage_degree(test_lesion, lesion_network, type='both')
print(f'The weights of input ROI/Lesion:  {damage_intensity}')
print(f'The percent of input ROI/Lesion:  {damage_percent} %')
## end. author@kangwu
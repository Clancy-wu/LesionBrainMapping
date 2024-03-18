#!/bin/python
## PALM analysis
## Reference: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM
#################################################
# Step 1: plam analysis.
#################################################
##  Definition Area
##  Reference: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/GLM
##  to know how to create contrast.txt
#################################################
palm_analysis_dir = '/home/clancy/data/LesionBrainMapping/output_test/palm_analysis'
contrast_matrix_file = '/home/clancy/data/LesionBrainMapping/output_test/palm_analysis/contrast.txt'
#################################################
##  Run
#################################################
from brainmapping.functions import run_palm
run_palm(palm_analysis_dir, contrast_matrix_file)
# Usage: run_palm(palm_analysis_dir, contrast_matrix_file, iterations=1000, save_1p=True,
#                 logp=False, tfce=False, two_tailed_flag=True, corrcon_flag=False, fdr_flag=False)
# iterations = 1000       # permutation iterations, default is 1000
# save_1p=True            # Save p values as 1 - p
# logp=False              # Save the output p-values as -log10(p)
# tfce=False              # Generate tfce output
# two_tailed_flag=True    # Run as two tailed test
# corrcon_flag=False      # Apply FWER-correction of p-values over multiple contrasts.
# fdr_flag=False          # Produce FDR-adjusted p-values.

# end. author@kangwu
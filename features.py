# main imports
import numpy as np
import sys

# config and modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

def compute_feature(feature_choice, samples):

    data = []

    if feature_choice == 'samples':
        data = samples

    if feature_choice == 'variances':

        incr_samples = []
        
        # evolution of variance
        for sample in samples:
            incr_samples.append(sample)
            data.append(np.var(incr_samples))

    return data
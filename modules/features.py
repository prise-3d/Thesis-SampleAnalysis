from . import config as config

import numpy as np

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
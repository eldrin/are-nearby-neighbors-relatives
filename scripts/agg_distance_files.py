from os.path import join, basename, dirname
import glob
import pickle as pkl
import pandas as pd
import numpy as np


data_root = '/tudelft.net/staff-bulk/ewi/insy/MMC/jaykim/datasets/nmd_valid_dists/msdlastfm50/'


# loading and aggregating
for alg in ['DTW', 'MCKL', 'SIMPLE']:

    dists = []
    for fn in glob.glob(join(data_root, alg, '*.pkl')):
        with open(fn, 'rb') as f:
            dists.extend(pkl.load(f))
    dists = np.array(dists)

    # save to a file
    np.save(join(data_root, '{}.npy'.format(alg)), dists)

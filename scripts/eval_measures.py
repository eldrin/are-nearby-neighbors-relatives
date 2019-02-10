import os
from os.path import join, abspath, basename, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm


from musiclatentconsistency.config import Config as cfg
from musiclatentconsistency.measures import (within_space_error,
                                             between_space_consistency_spearman)
from musiclatentconsistency.utils import load_distance_data


Dx, x_metadata = load_distance_data(
    '/mnt/data/nmd_dist/msdlastfm50/DTW.npy',
    '/home/ubuntu/workbench/music-purterbation-distance/dist_calc_msd_1k.txt'
)
within_X = within_space_error(Dx, x_metadata, verbose=True)
print(1 - within_X.groupby(['transform', 'magnitude'])['error'].mean())


Dz, z_metadata = load_distance_data(
    '/mnt/data/nmd_dist/AutoTagging_Z.npy',
    '/home/ubuntu/workbench/music-purterbation-distance/target_audio_npy.txt',
    domain='latent',
    latent_metric='euclidean'
)
within_Z = within_space_error(Dz, z_metadata, verbose=True)
print(1 - within_Z.groupby(['transform', 'magnitude'])['error'].mean())


between_XZ = between_space_consistency_spearman(
                Dx, Dz, x_metadata, z_metadata, verbose=True)
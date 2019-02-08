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
from musiclatentconsistency.measures import within_space_consistency
from musiclatentconsistency.utils import load_distance_data


data = load_distance_data('/mnt/data/nmd_dist/msdlastfm50/MCKL.npy', n_originals=1000)

within_X = []
pert_name_dict = {'PS':0, 'TS':1, 'PN':2, 'EN':3, 'MP':4}
n_perts = sum([len(mags) for name, mags in cfg.PERTURBATIONS])
with tqdm(total=n_perts) as pbar:
    for pert, magn in cfg.PERTURBATIONS:

        for mag in magn:
            pbar.update(1)
            filtered_data = data[
                (data['from_perturbation'] == pert_name_dict[pert]) &
                (data['from_magnitude'] == mag)
            ]

            faults = within_space_consistency(filtered_data, sample=1)
            for target, fault in faults.items():
                # put one row
                within_X.append({
                    'item': target,
                    'perturbation': pert,
                    'magnitude': mag,
                    'fault': fault
                })

# convert to the data to DataFrame
within_X = pd.DataFrame(within_X)

# calculate fault-rate
fault_rate = (within_X[~within_X.fault.isna()]
              .groupby(['perturbation', 'magnitude'])['fault']
              .apply(np.mean)
              .reset_index())
fault_rate = fault_rate.append([{'perturbation':'PS', 'magnitude':0, 'fault':0}])
fault_rate = fault_rate.append([{'perturbation':'TS', 'magnitude':1, 'fault':0}])



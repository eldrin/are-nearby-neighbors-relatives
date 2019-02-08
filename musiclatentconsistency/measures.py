import pandas as pd
import numpy as np


def within_space_consistency(filtered_data, sample=None, verbose=False):
    """"""
    d = filtered_data.sample(frac=sample)  # aliasing
    d = d[d['to_perturbation'] == 'original']
    
    output = {}
    targets = d['from'].unique()
    if verbose:
        targets = tqdm(targets, ncols=80)
        
    for target in targets:
        try:
            # retrieve target
            d_t = d[d['from'] == target]

            # get comparisons
            d_ts_s = d_t[d_t['to'] == target]['distance'].values
            if len(d_ts_s) == 0:
                raise ValueError()
            d_ts_s_prime = d_t[d_t['to'] != target]['distance'].values

            # save error
            output[target] = not np.all(d_ts_s < d_ts_s_prime)

        except Exception as e:
            output[target] = np.NaN
        
    return output


def between_space_consistency(D_x_t, D_z_t, mode='spearman', verbose=False):
    """Consistency between the two space
    
    Args:
        D_x_t (pd.DataFrame): contains the distance info from all 
                              transformations to original on original
                              data (x) domain
        D_z_t (pd.DataFrame): same with above data, on embedding domain (z)
        mode (str): flag for selecting measure mode {'spearman', 'accuracy'}
        verbose (bool): flag to indicate verbosity
    
    Returns:
        pd.DataFrame: between-space inconsistency
    """
    transformations = D_x_t['to_perturbation'].unique()
    targets = D_x_t['from'].unique()
    
    for t in transformations:
        d_x_t = D_x_t[D_x_t['to_perturbation'] == t]
        d_z_t = D_z_t[D_z_t['to_perturbation'] == t]
        
        for target in targets:
            # retrieve target
            d_x_t_ = d_x_t[d_x_t['to'] != target]['distance']
            d_z_t_ = d_z_t[d_z_t['to'] != target]['distance']
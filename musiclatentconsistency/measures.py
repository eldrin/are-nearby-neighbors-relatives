import pandas as pd
import numpy as np
from scipy import sparse as sp
from scipy.stats import spearmanr
from tqdm import tqdm

from .config import Config as cfg
from .utils import to_dense


def within_space_error(D, metadata, verbose=False):
    """Compute within-space consistency
    
    Args:
        D (sp.csr_matrix): contains pair-wise distance
        metadata (pd.DataFrame): describing details about distance data
        verbose (bool): verbosity flag
    
    Returns:
        pd.DataFrame: contains error info   
    """
    # filter out originals only
    originals = metadata[metadata['transform'] == 'original']
    result = []  # main container for output
    it = originals.index
    if verbose:
        it = tqdm(it, ncols=80)
        
    for s in it:
        # get idx of originals of others
        s_id = metadata.loc[s, 'audio_id']
        others = originals[originals['audio_id'] != s_id]
        mdata_ = metadata[metadata['audio_id'] == s_id]

        # for each perturbation and magnitude, get error
        for pert, magn in cfg.PERTURBATIONS:
            targets = mdata_[mdata_['transform'] == pert]

            for mag in magn:
                targets_ = targets[targets['magnitude'] == mag]

                # if any 'other' point is more close to the transformation
                # of the 'original' point, it'll be considered as error
                d_ts_s = to_dense(D[targets_.index, s]).ravel()
                d_ts_s_prime = to_dense(D[targets_.index, others.index]).ravel()
                error = 0 if np.all(d_ts_s < d_ts_s_prime) else 1

                # register to output container
                result.append({
                    'transform': pert,
                    'magnitude': mag,
                    'audio_index': s,
                    'error': error
                })

    return pd.DataFrame(result)


def between_space_consistency_spearman(
        Dx, Dz, x_metadata, z_metadata, verbose=False):
    """Consistency between the two space
    
    Args:
        Dx (pd.DataFrame): contains the distance info from all 
                              transformations to original on original
                              data (x) domain
        Dz (pd.DataFrame): same with above data, on embedding domain (z)
        x_metadata (pd.DataFrame): metadata for domain X
        z_metadata (pd.DataFrame): metadata for domain Z
        verbose (bool): verbosity flag
    
    Returns:
        pd.DataFrame: between-space inconsistency
    """
    result = []
    orig_x = x_metadata[x_metadata['transform'] == 'original']
    orig_z = z_metadata[z_metadata['transform'] == 'original']
    orig_indices = (
        orig_x
        .reset_index()
        .merge(
            orig_z.reset_index(),
            on='audio_id'
        )
    )[['index_x', 'index_y']]
    ix_orig_x = orig_indices.index_x.values.tolist()
    ix_orig_z = orig_indices.index_y.values.tolist()

    it = cfg.PERTURBATIONS + [('original', [0])]
    if verbose: it = tqdm(it, ncols=80) 
    for pert, magn in it:
        x_pert = x_metadata[x_metadata['transform'] == pert]
        z_pert = z_metadata[z_metadata['transform'] == pert]
        
        for mag in magn:
            x_pert_ = x_pert[x_pert['magnitude'] == mag]
            z_pert_ = z_pert[z_pert['magnitude'] == mag]
            
            indices = x_pert_.reset_index().merge(
                z_pert_.reset_index(), on='audio_id'
                )[['index_x', 'index_y']]
            
            ix_x = indices.index_x.values.tolist()
            ix_z = indices.index_y.values.tolist()
            dx = np.array(Dx[ix_x].todense())[:, ix_orig_x].ravel()
            dz = np.array(Dz[ix_z].todense())[:, ix_orig_z].ravel()
            
            result.append({
                'transform':pert,
                'magnitude': mag,
                'consistency_rho': spearmanr(dx, dz).correlation
            })

    return pd.DataFrame(result)


def between_space_consistency_accuracy(within_X_error, within_Z_error):
    """Consistency between the two space
    
    Args:
        within_X_error (pd.DataFrame): within-space error for data domain
        within_Z_error (pd.DataFrame): within-space error for latent domain
    
    Returns:
        pd.DataFrame: between-space inconsistency
    """
    wX = within_X_error.set_index(['audio_index', 'magnitude', 'transform'])
    wZ = within_Z_error.set_index(['audio_index', 'magnitude', 'transform'])
    w = wX.join(wZ, lsuffix='_x', rsuffix='_z').reset_index()
    
    C_acc = (
        w.groupby(['transform', 'magnitude'])
        .apply(lambda x: np.mean(x['error_x'] == x['error_z']))
        .reset_index()
    )
    C_acc.columns = ['transform', 'magnitude', 'consistency_acc'] 
    
    return C_acc
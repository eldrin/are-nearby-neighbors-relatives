import os
from os.path import join, abspath, basename, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from functools import partial
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from musiclatentconsistency.config import Config as cfg
from musiclatentconsistency.measures import (within_space_error,
                                             between_space_consistency_spearman)
from musiclatentconsistency.utils import load_distance_data, parse_metadata


def visualize_data(total_data, value_key='accuracy', hue=None):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, transform in zip(axs, ['PS', 'TS', 'PN', 'EN', 'MP']):
        data = total_data[total_data['transform'] == transform].copy()
        og_data = total_data[total_data['transform'] == 'OG'].copy()
        og_data.loc[:, 'transform'] = transform
        if transform in {'PN', 'EN'}:
            og_data.loc[:, 'magnitude'] = 48
        elif transform == 'MP':
            og_data.loc[:, 'magnitude'] = 400
        elif transform == 'TS':
            og_data.loc[:, 'magnitude'] = 1

        data = pd.concat([data, og_data])
        sns.lineplot(data=data, x='magnitude', y=value_key, hue=hue, ax=ax)
        ax.set_title(transform)

        
def get_within_X(dist='SIMPLE', include_original=False):
    data_root = '/home/ubuntu/workbench/music-purterbation-distance/data/'
    dist_root = '/mnt/data/nmd_dist/msdlastfm50/'
    
    if dist == 'MCKL':
        hi_metadata_fn = join(data_root, 'dist_calc_msd_1k_gmm_high_res_87.txt')
        lo_metadata_fn = join(data_root, 'dist_calc_msd_1k.txt')
    else:
        hi_metadata_fn = join(data_root, 'dist_calc_msd_1k_high_res_87.txt')
        lo_metadata_fn = join(data_root, 'dist_calc_msd_1k.txt')
        
    # hi resolution
    Dx_hi, x_metadata_hi = load_distance_data(
        join(dist_root, '{}_hi_res.npy'.format(dist)), hi_metadata_fn
    )
    within_X_hi = within_space_error(
        Dx_hi, x_metadata_hi, 
        cfg.PERTURBATIONS_HI_RES, verbose=True
    )
    within_X_hi['consistency'] = within_X_hi['error'].apply(lambda x: 1 - x)

    # low resolution
    Dx_lo, x_metadata_lo = load_distance_data(
        join(dist_root, '{}.npy'.format(dist)), lo_metadata_fn
    )
    within_X_lo = within_space_error(
        Dx_lo, x_metadata_lo, 
        cfg.PERTURBATIONS, verbose=True
    )
    within_X_lo['consistency'] = within_X_lo['error'].apply(lambda x: 1 - x)

    if include_original:
        data = pd.concat([within_X_hi, within_X_lo])
    else:
        data = pd.concat([
            within_X_hi[within_X_hi['transform'] != 'OG'],
            within_X_lo[within_X_lo['transform'] != 'OG']
        ])
        
    # visualization
    visualize_data(data, value_key='consistency')

    return data


def get_within_Z(z_fn, z_metadata_fn, dist='euclidean',
                 include_original=False, visualize=False):
    Dz, z_metadata = load_distance_data(
        z_fn, z_metadata_fn, domain='latent', latent_metric=dist,
        header=True
    )
    within_Z = within_space_error(Dz, z_metadata, cfg.PERTURBATIONS_HI_LOW, verbose=True)
    within_Z['consistency'] = within_Z['error'].apply(lambda x: 1 - x)
    
    if not include_original:
        data = within_Z[within_Z['transform'] != 'OG']
    
    if visualize:
        # visualization
        visualize_data(data, value_key='consistency')
    
    return data


def load_Dx(x_dist, data_root, dist_root):
    # data_root = '/home/ubuntu/workbench/music-purterbation-distance/data/'
    # dist_root = '/mnt/data/nmd_dist/msdlastfm50/'

    if x_dist == 'MCKL':
        hi_metadata_fn = join(data_root, 'dist_calc_msd_1k_gmm_high_res_87.txt')
        lo_metadata_fn = join(data_root, 'dist_calc_msd_1k.txt')
    else:
        hi_metadata_fn = join(data_root, 'dist_calc_msd_1k_high_res_87.txt')
        lo_metadata_fn = join(data_root, 'dist_calc_msd_1k.txt')

    # hi resolution
    Dx_hi, x_metadata_hi = load_distance_data(
        join(dist_root, '{}_hi_res.npy'.format(x_dist)), hi_metadata_fn
    )
    # low resolution
    Dx_lo, x_metadata_lo = load_distance_data(
        join(dist_root, '{}.npy'.format(x_dist)), lo_metadata_fn
    )
    
    orig_hi = x_metadata_hi[x_metadata_hi['transform'] == 'OG']
    orig_lo = x_metadata_lo[x_metadata_lo['transform'] == 'OG']
    orig_indices = (
        orig_hi
        .reset_index()
        .merge(
            orig_lo.reset_index(),
            on='audio_id'
        )
    )[['index_x', 'index_y']]
    ix_orig_hi = orig_indices.index_x.values.tolist()
    ix_orig_lo = orig_indices.index_y.values.tolist()
    
    # join the two Dx with same audio_id order
    Dx = sp.vstack([Dx_hi[:, ix_orig_hi], Dx_lo[1000:, ix_orig_lo]])
    x_metadata = pd.concat([x_metadata_hi, x_metadata_lo.iloc[1000:]])
    x_metadata['audio_id'] = x_metadata['audio_id'].values.astype(int)
    x_metadata = x_metadata.reset_index().drop('index', axis=1)
    
    return Dx, x_metadata


def get_between_XZ(z_fn, z_metadata_fn, type,
                   dist_x=None, dist_z=None, Dx=None, x_metadata=None,
                   verbose=True):
    if type == 'corr':
        if (Dx is None) and (x_metadata is None):
            Dx, x_metadata = load_Dx(dist_x)
        
        # load data for Z
        Dz, z_metadata = load_distance_data(
            z_fn, z_metadata_fn, domain='latent', latent_metric=dist_z,
            header=True
        )

        between_XZ_consistency = between_space_consistency_spearman(
            Dx, Dz, x_metadata, z_metadata,
            perturbations=cfg.PERTURBATIONS_HI_LOW,
            verbose=verbose
        )
        
    else:
        
        within_X = get_within_X(dist_x)
        within_Z = get_within_Z(z_fn, z_metadata_fn, dist_z)

        wX = within_X.set_index(['audio_index', 'magnitude', 'transform'])
        wZ = within_Z.set_index(['audio_index', 'magnitude', 'transform'])
        w = wX.join(wZ, lsuffix='_x', rsuffix='_z').reset_index()
        between_XZ_consistency = w.groupby(
            ['transform', 'magnitude']
        ).apply(lambda x: np.mean(x['error_x'] == x['error_z'])).reset_index()
        between_XZ_consistency.columns = ['transform', 'magnitude', 'consistency_acc']
    
    return between_XZ_consistency




if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_distance_root", type=str,
                        help="path where audio distance saved")
    parser.add_argument("audio_metadata_root", type=str,
                        help="path where audio metadata saved")
    parser.add_argument("latent_data_root", type=str,
                        help="data path where latent points saved")
    parser.add_argument("task", type=str, choices={'AT', 'AE', 'IR', 'VS'},
                        help="type of the task of which the model is trained")
    parser.add_argument("trial", type=int, choices={0, 1, 2, 3, 4}, help="trial of model")
    parser.add_argument("cw_type", type=str, default='corr', choices={'corr', 'accuracy'},
                        help="consistency type")
    parser.add_argument("dist_x", type=str, choices={'DTW', 'SIMPLE', 'MCKL'},
                        help="distance metric on audio domain")
    parser.add_argument("dist_z", type=str, choices={'euclidean', 'cosine'},
                        help="distance metric on latent domain")
    parser.add_argument("out_path", help='filename to dump latent points and metadata')
    args = parser.parse_args() 

    root = join(args.latent_data_root, '{}/'.format(args.task))
    Dx, x_metadata = load_Dx(dist_x, args.audio_distance_root, args.audio_metadata_root)
    bet_XZ = get_between_XZ(
        z_fn=join(root, '{}_z_trial{:d}.npy'.format(args.task, args.trial)),
        z_metadata_fn=join(root, '{}_z_trial{:d}.csv'.format(args.task, args.trial)),
        type=args.cw_type, dist_z=args.dist_z, dist_x=args.dist_x,
        Dx=Dx, x_metadata=x_metadata, verbose=True
    )
    bet_XZ['task'] = task
    bet_XZ['trial'] = trial
    bet_XZ['d_x'] = dist_x
    bet_XZ['d_z'] = dist_z
    bet_XZ['type'] = c_type
    
    # save
    bet_XZ.to_csv(join(
        args.out_path,
        '{}_{:d}_{}_{}_{}.csv'.format(
            args.task, args,trial, args.cw_type,
            args.dist_x, args.dist_z
    ))
from os.path import join, basename, dirname
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.spatial.distance import cdist

from .config import Config as cfg


def load_distance_data(data_fn, filelist_fn, domain='input',
                       latent_metric='euclidean'):
    """Load and convert data to pandas DataFrame to ease of analysis
    
    Args:
        data_fn (str): path to the data.
                       For input domain's case, contains list of triplets (i, j, d)
                       For latent domain's case, contains latent points (n_points, dim)
        filelist_fn (str): path to the meta-data. contains list of filenames
        domain (str): flag for indicating which data domain is target
                      {'input', 'latent'}
        latent_metric (str): distance metric used to calculate distance between
                             latent points. only used when domain == 'latent'
                             {'euclidean', 'cosine'}
                      
    Returns:
        sp.csr_matrix: (partial) distance matrix
        pd.DataFrame: contains metadata 
    """ 
    # load metadata
    with open(filelist_fn, 'r') as f:
        metadata = parse_metadata(f.readlines())
    
    # load distance data
    if domain == 'input':
        D = np.load(data_fn)
        D = sp.coo_matrix(
            (D[:, 2], (D[:, 0].astype(int), D[:, 1].astype(int))),
        ).tocsr()
        
    elif domain == 'latent':
        Z = np.load(data_fn)        
        D = calc_latent_dist(Z, metadata, metric=latent_metric)
        D = sp.coo_matrix(D).tocsr()

    else:
        raise ValueError("[ERROR] only 'input' and 'latent' is supportd!")
        
    return D, metadata


def parse_metadata(fns):
    """Parse filenames to build metadata 
    
    Args:
        fns (list): a list that contains filenames
        
    Returns:
        pd.DataFrame: a structured table contains parsed info
    """
    def parse_base(fn):
        x = basename(fn.replace('\n', ''))
        x = x.split('.npy')[0]
        x = x.split('_')
        return x
    
    data = []
    for fn, row in zip(fns, map(parse_base, fns)):
        d = {'audio_id': row[0], 'start': row[1], 'end': row[2]}
        if len(row) < 5:
            d.update({'transform': 'original', 'magnitude': 0})
        else:
            d.update({'transform': row[3],
                      'magnitude': float(row[4][1:-1])})
        d.update({'fn': fn})
        data.append(d)

    return pd.DataFrame(data)


def calc_latent_dist(Z, metadata, metric='euclidean'):
    """Calculate distance (as same as X domain) between point in Z
    
    Args:
        Z (np.ndarray): input n-dimensional points (n_points, n_dim)
        metadata (pd.DataFrame): contains metadata for each point
    
    Returns:
        sp.csr_matrix: resulted data
    """
    originals = metadata[metadata['transform'] == 'original']
    mask = np.zeros(Z.shape[0], dtype=bool)
    mask[originals.index] = True
    z_original = Z[mask]
    return cdist(z_original, Z, metric=metric).T
    

def to_dense(csr_mat):
    """Convert csr matrix to dense ndarray
    
    Args:
        csr_mat (sp.csr_matrix): input sparse (slice) matrix
    
    Returns:
        np.ndarray: converted matrix
    """
    return np.array(csr_mat.todense())


def save_mulaw(fn, y, sr=22050, quantization_channel=256):
    """Save a signal encoded with Mu-Law

    Args:
        fn (str): path to save the signal
        y (numpy.ndarray): signal vector (n_samples,)
        sr (int): sampling rate
        quantization_channel (int): # of quantization channels
    """
    mu = quantization_channel - 1
    safe_audio_abs = np.minimum(np.abs(y), 1.)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(y) * magnitude
    y_ = ((signal + 1) / 2 * mu + 0.5).astype(np.uint8)
    np.save(fn, y_)


def load_mulaw(fn, quantization_channel=256):
    """Load a Mu-Law encoded signal by decoding it
    """
    signal = np.load(fn)
    mu = quantization_channel - 1
    signal = 2 * (signal / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

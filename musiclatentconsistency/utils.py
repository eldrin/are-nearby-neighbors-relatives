import numpy as np
import pandas as pd

from .config import Config as cfg


def load_distance_data(fn, n_originals, perturbations=cfg.PERTURBATIONS):
    """Load and convert data to pandas DataFrame to ease of analysis
    
    Args:
        fn (str): path to the data. contains list of triplets (i, j, d)
    """
    # load the dataset
    data = pd.DataFrame(
        np.load(fn), columns=['from', 'to', 'distance']
    )
    
    # get idx-perturbation types / magnitude map
    pert_mags = [0] * n_originals
    pert_types = ['original'] * n_originals
    pert_name_dict = {'PS':0, 'TS':1, 'PN':2, 'EN':3, 'MP':4}
    for _ in range(n_originals):
        for p_name, p_mags in perturbations:
            for p_mag in p_mags:
                pert_mags.append(p_mag)
                # pert_types.append(p_name)
                pert_types.append(pert_name_dict[p_name])
    pert_mags = dict(enumerate(pert_mags))
    pert_types = dict(enumerate(pert_types))
    n_perts = sum([len(mags) for name, mags in perturbations])

    # get idx-original_idx map
    rawidx2orgidx = {}
    for i in range(n_originals):
        for j in range(n_perts):
            rawidx2orgidx[n_originals + i * n_perts + j] = i
        rawidx2orgidx[i] = i
        
    # applying maps to generate full dataset
    data.loc[:, 'from_perturbation'] = data['from'].map(pert_types)
    data.loc[:, 'from_magnitude'] = data['from'].map(pert_mags)
    data.loc[:, 'to_perturbation'] = data['to'].map(pert_types)
    data.loc[:, 'to_magnitude'] = data['to'].map(pert_mags)
    data.loc[:, 'from_original_idx'] = data['from'].map(rawidx2orgidx)
    data.loc[:, 'to_original_idx'] = data['to'].map(rawidx2orgidx)
    
    # drop unnecessary cols
    data.drop(['from', 'to'], axis=1, inplace=True)
    data.rename(
        index=str,
        columns={"from_original_idx": "from", "to_original_idx": "to"},
        inplace=True
    )
    
    return data


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

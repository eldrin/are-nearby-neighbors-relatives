import copy # edit in place
import numpy as np
from scipy.signal import lfilter

from .config import Config as cfg

"""
These functions are borrowed from https://github.com/BrechtDeMan/loudness.py

Brecht De Man, "Evaluation of Implementations of the EBU R128 Loudness Measurement," 145th International Convention of the Audio Engineering Society, October 2018.
""" 

# pre-filter 1
f0 = 1681.9744509555319
G  = 3.99984385397
Q  = 0.7071752369554193
# TODO: precompute
K  = np.tan(np.pi * f0 / cfg.FS) 
Vh = np.power(10.0, G / 20.0)
Vb = np.power(Vh, 0.499666774155)
a0_ = 1.0 + K / Q + K * K
b0 = (Vh + Vb * K / Q + K * K) / a0_
b1 = 2.0 * (K * K -  Vh) / a0_
b2 = (Vh - Vb * K / Q + K * K) / a0_
a0 = 1.0
a1 = 2.0 * (K * K - 1.0) / a0_
a2 = (1.0 - K / Q + K * K) / a0_
FILTER_COEF1 = [b0,b1,b2], [a0,a1,a2]

f0 = 38.13547087613982
Q  = 0.5003270373253953
K  = np.tan(np.pi * f0 / cfg.FS)
a0 = 1.0
a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
b0 = 1.0
b1 = -2.0
b2 = 1.0
FILTER_COEF2 = [b0,b1,b2], [a0,a1,a2]


def lufs_norm(y, ref=None, fs=22050):
    if ref is None:
        lufs_ref = -14
    else:
        lufs_ref = calculate_loudness(ref, fs)
        
    lufs_y = calculate_loudness(y, fs)
    coef = np.float32(10**((lufs_ref - lufs_y) / 20))
    return y * coef


def calculate_loudness(signal, fs=cfg.FS, G = [1.0, 1.0, 1.0, 1.41, 1.41]):
    # filter
    if len(signal.shape)==1: # if shape (N,), then make (N,1)
        signal_filtered = copy.copy(signal.reshape((signal.shape[0],1)))
    else:
        signal_filtered = copy.copy(signal)
        
    for i in range(signal_filtered.shape[1]):
        signal_filtered[:,i] = K_filter(signal_filtered[:,i])

    # mean square
    T_g = 0.400 # 400 ms gating block
    Gamma_a = -70.0 # absolute threshold: -70 LKFS
    overlap = .75 # relative overlap (0.0-1.0)
    step = 1 - overlap

    T = signal_filtered.shape[0]/fs # length of measurement interval in seconds
    j_range = np.arange(0,int((T-T_g)/(T_g*step)))
    z = np.ndarray(shape=(signal_filtered.shape[1],len(j_range)))
    # write in explicit for-loops for readability and translatability
    for i in range(signal_filtered.shape[1]): # for each channel i
        for j in j_range: # for each window j
            lbound = int(np.round(fs*T_g*j*step).astype(int))
            hbound = int(np.round(fs*T_g*(j*step+1)).astype(int))
            z[i,j] = (1/(T_g*fs))*np.sum(np.square(signal_filtered[lbound:hbound, i]))

    G_current = np.array(G[:signal_filtered.shape[1]]) # discard weighting coefficients G_i unused channels
    n_channels = G_current.shape[0]
    l = [-.691 + 10.0*np.log10(np.sum([G_current[i]*z[i,j.astype(int)] for i in range(n_channels)])) \
             for j in j_range]

    # throw out anything below absolute threshold:
    indices_gated = [idx for idx,el in enumerate(l) if el > Gamma_a] 
    z_avg = [np.mean([z[i,j] for j in indices_gated]) for i in range(n_channels)]
    Gamma_r = -.691 + 10.0*np.log10(np.sum([G_current[i]*z_avg[i] for i in range(n_channels)])) - 10.0
    # throw out anything below relative threshold:
    indices_gated = [idx for idx,el in enumerate(l) if el > Gamma_r] 
    z_avg = [np.mean([z[i,j] for j in indices_gated]) for i in range(n_channels)]
    L_KG = -.691 + 10.0*np.log10(np.sum([G_current[i]*z_avg[i] for i in range(n_channels)]))

    return L_KG

def K_filter(signal):
    global FILTER_COEF1, FILTER_COEF2
    
    # apply K filtering as specified in EBU R-128 / ITU BS.1770-4
    signal_1 = lfilter(FILTER_COEF1[0], FILTER_COEF1[1], signal)
    
    # pre-filter 2
    signal_2 = lfilter(FILTER_COEF2[0], FILTER_COEF2[1], signal_1) 
    
    return signal_2 # return signal passed through 2 pre-filters
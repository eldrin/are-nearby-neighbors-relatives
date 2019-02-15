import os
from os.path import join, abspath, basename, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-distances'))

import argparse
from functools import partial
import pickle as pkl

import numpy as np

from musiclatentconsistency.utils import save_mulaw, load_mulaw
from audiodistances.utils import parmap


def _random_crop_signal(y, n_samples, sig_len):
    """Randomly crop given signal with given N chunks
    
    Args:
        y (np.ndarray): signal to be cropped
        n_samples (int): desired nbr of chunks
        sig_len (int): desired length of cropped signal
    
    Returns:
        tuple: contains
                (1. cropped signal
                 2. starting index)
    """
    outputs = []
    if n_samples == 1:
        if len(y) < sig_len:
            # if the input signal is shorter than desired length,
            # pad it (both left and right side)
            start = 0
            to_pad = sig_len - len(y)
            pad = np.zeros((int(to_pad / 2),), dtype=y.dtype)
            y_ = np.r_[pad, y, pad]

        else:
            # randomly crop the signal with given length
            start = np.random.choice(len(y) - sig_len)
            y_ = y[start:start + sig_len]
            
        # save result & register information
        outputs.append((y_, start))
        
    elif n_samples > 1:
        # randomly crop N clips
        # 1. split signal to N even chunks
        l = int(len(y) / n_samples)  # length of each chunk
        bounds = [(l*n, l*(n + 1)) for n in range(n_samples)]

        for start, end in bounds:
            # slice chunk of interest
            y_ = y[start:end]
            
            # get within-chunk starting point
            start_ = np.random.choice(len(y_) - sig_len)
            
            # save cropped signal & register info
            start_raw = start_ + start
            outputs.append(
                (y_[start_: start_ + sig_len], start_raw)
            )
    else:
        raise ValueError(
            '[ERROR] only positive integers are available for n_samples!'
        )
        
    return outputs


def _random_crop(in_fn, out_root, sig_len, n_samples=1, sr=22050):
    """Randomly drop audio and save

    Args:
        in_fn (str): path to load the original signal
        out_root (str): path to save the cropped signals
        sig_len (int): length of the cropped signal
        n_samples (int): the number of clips to be cropped
        sr (int): sampling rate of the signal
    
    Returns:
        str: path where the file is saved
        tuple: contains starting point and the end point
    """
    # helper function to save
    def _save(signal, sample_rate, start_ix):
        out_fn = join(
            out_root,
            splitext(basename(in_fn))[0] +
            '_{:d}_{:d}.npy'.format(
                int(start_ix), int(start_ix+sig_len)
            )
        )
        save_mulaw(out_fn, signal, sample_rate)
        return (out_fn, (start_ix, start_ix + sig_len))
        
        
    if basename(in_fn).split('.')[-1] == 'npy':
        y = load_mulaw(in_fn)
    else:
        y, sr = librosa.load(in_fn, sr=sr)
        
    # check nbr of chunks is too many considering signal length
    assert sig_len * n_samples > len(y)
    
    info = [_save(sig, sr, st)
            for sig, st
            in _random_crop_signal(y, n_samples, sig_len)]
    
    return info


def crop_signals(fns, out_root, n_samples, sr=22050, sig_len=44100, n_jobs=1):
    """Extract MFCCs

    Args:
        fns (str): file name of the music
        out_root (str): path to dump files
        n_samples (int): number of chunks
        sr (int): sampling rate of the signal
        sig_len (int): length of the signal 
        n_jobs (int): number of parallel jobs
    """
    info = parmap(
        partial(_random_crop, out_root=out_root,
                sig_len=sig_len, n_samples=n_samples, sr=sr),
        fns, n_workers=n_jobs, verbose=True
    )
    with open(join(out_root, 'crop_info.pkl'), 'wb') as f:
        pkl.dump(dict(info), f)
    

if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_files",
                        help='text file contains all the file names of music')
    parser.add_argument("out_path", help='path to dump output files')
    parser.add_argument("sig_len", type=int, default=44100, help='desired length of the cropped signal')
    parser.add_argument("n_samples", type=int, default=1, help='desired number of chunks to crop')
    parser.add_argument("--sr", type=int, default=22050, help='sample rate of the signal')
    parser.add_argument("--n-jobs", type=int, default=1, help='number of parallel jobs')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # adjust crop length for `speed-up` transformation
    sig_len = args.sig_len * 2

    # process!
    crop_signals(fns, args.out_path, args.n_samples,
                 args.sr, sig_len, n_jobs=args.n_jobs)
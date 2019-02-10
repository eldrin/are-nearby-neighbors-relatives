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


def _random_crop(in_fn, out_root, sig_len, sr=22050):
    """Randomly drop audio and save

    Args:
        in_fn (str): path to load the original signal
        out_root (str): path to save the cropped signals
        sig_len (int): length of the cropped signal
        sr (int): sampling rate of the signal
    """
    if basename(in_fn).split('.')[-1] == 'npy':
        y = load_mulaw(in_fn)
    else:
        y, sr = librosa.load(in_fn, sr=sr)
        
    # if the input signal is shorter than desired length,
    # pad it (both left and right side)
    if len(y) < sig_len:
        start = 0
        to_pad = sig_len - len(y)
        pad = np.zeros((int(to_pad / 2),), dtype=y.dtype)
        y_ = np.r_[pad, y, pad]
        
    else:
        # randomly crop the signal with given length
        start = np.random.choice(len(y) - sig_len)
        y_ = y[start:start + sig_len]

    out_fn = join(
        out_root,
        splitext(basename(in_fn))[0] +
        '_{:d}_{:d}.npy'.format(int(start), int(start+sig_len))
    )
    save_mulaw(out_fn, y_, sr)
    return (out_fn, (start, start + sig_len))


def crop_signals(fns, out_root, sr=22050, sig_len=44100, n_jobs=1):
    """Extract MFCCs

    Args:
        fns (str): file name of the music
        out_root (str): path to dump files
        sr (int): sampling rate of the signal
        sig_len (int): length of the signal 
        n_jobs (int): number of parallel jobs
    """
    info = parmap(
        partial(_random_crop, out_root=out_root, sig_len=sig_len, sr=sr),
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
    parser.add_argument("--sr", type=int, default=22050, help='sample rate of the signal')
    parser.add_argument("--n-jobs", type=int, default=1, help='number of parallel jobs')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # adjust crop length for `speed-up` transformation
    sig_len = args.sig_len * 2

    # process!
    crop_signals(fns, args.out_path, args.sr, sig_len, n_jobs=args.n_jobs)

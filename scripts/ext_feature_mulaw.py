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
import librosa

from musiclatentconsistency.utils import save_mulaw, load_mulaw
from audiodistances.utils import parmap


def _extract_mfcc(fn, out_root, include_coef0=False,
                  sig_len=44100, sr=22050, n_mfccs=25):
    """Helper function to extract MFCC from a single music file

    Args:
        fn (str): filename for a single music file
        out_root (str): path to dump output MFCCs
        include_coef0 (bool): decides including 1st coefficient
        sig_len (int): length of desired signal
        sr (int): sampling rate
        n_mfcc (int): number of coefficients
    """
    if basename(fn).split('.')[-1] == 'npy':
        y = load_mulaw(fn)
    else:
        y, sr = librosa.load(fn, sr=sr)
        
    # we do not include the first coefficient
    M = librosa.feature.mfcc(y, sr, n_mfcc=n_mfccs).T
    if not include_coef0: M = M[:, 1:]
    out_fn = join(out_root, splitext(basename(fn))[0] + '.npy')
    np.save(out_fn, M)


def ext_mfccs(fns, out_root, n_mfccs=25,
              n_jobs=1, include_coef0=False):
    """Extract MFCCs

    Args:
        fns (str): file name of the music
        out_root (str): path to dump files
        n_mfcc (int): number of coefficients
        n_jobs (int): number of parallel jobs
        include_coef0 (bool): decides including 1st coefficient
    """
    parmap(
        partial(_extract_mfcc,
                out_root=out_root,
                include_coef0=include_coef0),
        fns, total=len(fns), n_workers=n_jobs, verbose=True
    )
    

if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_files",
                        help='text file contains all the file names of music')
    parser.add_argument("out_path", help='path to dump output files')
    parser.add_argument('--include-coef0', dest='coef0', action='store_true')
    parser.add_argument('--exclude-coef0', dest='coef0', action='store_false')
    parser.set_defaults(coef0=False)
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # process!
    ext_mfccs(fns, args.out_path, n_jobs=args.n_jobs,
              include_coef0=args.coef0)
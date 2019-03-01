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


def _extract_mfcc(fn, out_root, sig_len=44100, sr=22050, n_mfccs=25):
    """Helper function to extract MFCC from a single music file

    Args:
        fn (str): filename for a single music file
        out_root (str): path to dump output MFCCs
        sig_len (int): length of desired signal
        sr (int): sampling rate
        n_mfcc (int): number of coefficients
    """
    if basename(fn).split('.')[-1] == 'npy':
        y = load_mulaw(fn)
    else:
        y, sr = librosa.load(fn, sr=sr)
        
    # we do not include the first coefficient
    M = librosa.feature.mfcc(y, sr, n_mfcc=n_mfccs).T[:, 1:]
    out_fn = join(out_root, splitext(basename(fn))[0] + '.npy')
    np.save(out_fn, M)


def ext_mfccs(fns, out_root, n_mfccs=25, n_jobs=1):
    """Extract MFCCs

    Args:
        fns (str): file name of the music
        out_root (str): path to dump files
        n_mfcc (int): number of coefficients
        n_jobs (int): number of parallel jobs
    """
    parmap(
        partial(_extract_mfcc, out_root=out_root),
        fns, total=len(fns), n_workers=n_jobs, verbose=True
    )
    

if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_files",
                        help='text file contains all the file names of music')
    parser.add_argument("out_path", help='path to dump output files')
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # process!
    ext_mfccs(fns, args.out_path, n_jobs=args.n_jobs)

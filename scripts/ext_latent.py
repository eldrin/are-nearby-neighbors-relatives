import os
from os.path import join, abspath, basename, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'music-nn-models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-distances'))

import argparse
from functools import partial
import torch
import numpy as np
import librosa

from musiclatentconsistency.utils import save_mulaw, load_mulaw
from audiodistances.utils import parmap
from musicnn.models import VGGlike2DAutoTagger
from musicnn.datasets import TAGS


def _extract_latent(fn, model, sr=22050):
    """Helper function to extract latent embedding for a single music file

    Args:
        fn (str): filename for a single music file
        out_root (str): path to dump output MFCCs
        sr (int): sampling rate
    """
    if basename(fn).split('.')[-1] == 'npy':
        y = load_mulaw(fn)
    else:
        y, sr = librosa.load(fn, sr=sr)

    y = y.astype(np.float32)
    return model.E(model.preproc(torch.from_numpy(y)[None])).data.numpy()


def ext_latents(fns, out_fn, model, n_jobs=1):
    """Extract MFCCs

    Args:
        fns (str): file name of the music
        out_fn (str): path to dump files
        model (BaseModel): model used for the extraction
        n_jobs (int): number of parallel jobs
    """
    # process
    Z = parmap(
        partial(_extract_latent, model=model),
        fns, n_workers=n_jobs, verbose=True
    )
    # save the output
    np.save(out_fn, np.array(Z))


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_files",
                        help='text file contains all the file names of music')
    parser.add_argument("model_path", help='path to model checkpoint dump')
    parser.add_argument("out_path", help='path to dump output files')
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # load the model
    checkpoint = torch.load(args.model_path, lambda a, b: a)
    model = VGGlike2DAutoTagger(len(TAGS))
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])

    # process!
    ext_latents(fns, args.out_path, model, n_jobs=args.n_jobs)

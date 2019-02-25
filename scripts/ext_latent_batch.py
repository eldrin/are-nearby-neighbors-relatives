import os
from os.path import join, abspath, basename, dirname, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'music-nn-models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-distances'))

import argparse
from functools import partial
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from musiclatentconsistency.utils import save_mulaw, load_mulaw, parse_metadata
from musicnn.models import (VGGlike2DAutoTagger,
                            VGGlike2DAutoEncoder,
                            VGGlike2DUNet,
                            ShallowAutoTagger)
from musicnn.datasets.autotagging import TAGS
from musicnn.datasets.instrecognition import CLS

TASK_MODEL_MAP = {
    'auto_tagging': partial(VGGlike2DAutoTagger,
                            n_outputs=len(TAGS), layer1_channels=16),
    'auto_tagging_mfcc': partial(ShallowAutoTagger,
                                 n_outputs=len(TAGS), feat_dim=120),
    'inst_recognition': partial(VGGlike2DAutoTagger,
                                n_outputs=len(CLS), layer1_channels=16),
    'auto_encoder': partial(VGGlike2DAutoEncoder, layer1_channels=16),
    'source_separation': partial(VGGlike2DUNet, layer1_channels=16)
}


def _extract_latent(fn, model, sr=22050, is_mulaw=True):
    """Helper function to extract latent embedding for a single music file

    Args:
        fn (str): filename for a single music file
        sr (int): sampling rate
    """
    if basename(fn).split('.')[-1] == 'npy':
        if is_mulaw:
            y = load_mulaw(fn)
        else:
            y = np.load(fn)
    else:
        y, sr = librosa.load(fn, sr=sr)

    y = y.astype(np.float32)
    if len(y) > model.sig_len:
        # find the center and crop from there
        mid = int(len(y) / 2)
        half_len = int(model.sig_len / 2)
        start_point = mid - half_len
        y = y[start_point: start_point + model.sig_len]

    elif len(y) < model.sig_len:
        # zero-pad
        rem = model.sig_len - len(y)
        y = np.r_[y, np.zeros((rem,), dtype=y.dtype)]

    return model.get_bottleneck(
        torch.from_numpy(y)[None]
    ).data.numpy()[0]



def _mfcc(y, sr):
    """Calc MFCC feature

    Args:
        y (np.ndarray): signal (1d)
        sr (int): sampling rate

    Returns:
        np.ndarray: MFCC-based feature (6 * n_mfcc,)
    """
    # get MFCC vectors (t, n_mfcc)
    m = librosa.feature.mfcc(y, n_mfcc=40, sr=sr).T
    dm = m[1:] - m[:-1]
    ddm = dm[1:] - dm[:-1]

    # get stats
    feature = np.r_[
        m.mean(0), dm.mean(0), ddm.mean(0),
        m.std(0), dm.std(0), ddm.std(0)
    ]

    return feature  # (n_mfcc * 6,)


def _extract_mfcc(fn, sr=22050, is_mulaw=True):
    """Extract MFCC

    Args:
        fn (str): filename for a single music file
        sr (int): sampling rate

    Returns:
        np.ndarray: MFCC feature vector
    """
    if basename(fn).split('.')[-1] == 'npy':
        if is_mulaw:
            y = load_mulaw(fn)
        else:
            y = np.load(fn)
    else:
        y, sr = librosa.load(fn, sr=sr)
    return _mfcc(y, sr)


def ext_latents(fns, out_fn, model=None, n_jobs=1, is_mulaw=True, batch_sz=100):
    """Extract latent features

    Args:
        fns (str): file name of the music
        out_fn (str): path to dump files
        model (BaseModel, None, 'mfcc'): model used for the extraction
        n_jobs (int): number of parallel jobs
    """
    is_gpu = next(model.parameters()).is_cuda

    # if (model is None) or (model == 'mfcc'):
    #     f = partial(_extract_mfcc, is_mulaw=is_mulaw)
    # else:
    #     f = partial(_extract_latent, model=model, is_mulaw=is_mulaw)

    batch = []
    Z = []
    for i, fn in tqdm(enumerate(fns), total=len(fns), ncols=80):
        if basename(fn).split('.')[-1] == 'npy':
            if is_mulaw:
                y = load_mulaw(fn)
            else:
                y = np.load(fn)
        else:
            y, sr = librosa.load(fn, sr=sr)

        y = y.astype(np.float32)
        if len(y) > model.sig_len:
            # find the center and crop from there
            mid = int(len(y) / 2)
            half_len = int(model.sig_len / 2)
            start_point = mid - half_len
            y = y[start_point: start_point + model.sig_len]

        elif len(y) < model.sig_len:
            # zero-pad
            rem = model.sig_len - len(y)
            y = np.r_[y, np.zeros((rem,), dtype=y.dtype)]

        batch.append(y)

        if (len(batch) >= batch_sz) or ((i + 1) == len(fns)):
            batch = torch.from_numpy(np.array(batch).astype(np.float32))
            if is_gpu:
                batch = batch.cuda()

            z = model.get_bottleneck(batch)

            if is_gpu:
                z = z.data.cpu().numpy()
            else:
                z = z.data.numpy()
            Z.append(z)
            batch = []
    Z = np.concatenate(Z, axis=0)

    # save the output
    np.save(out_fn, np.array(Z))


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_files",
                        help='text file contains all the file names of music')
    parser.add_argument("task", type=str, default='autotagging',
                        choices=set(TASK_MODEL_MAP.keys()) | {'mfcc'},
                        help="type of the task of which the model is trained")
    parser.add_argument("model_path", help='path to model checkpoint dump')
    parser.add_argument("out_fn", help='filename to dump latent points and metadata')
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    parser.add_argument('--mulaw', dest='mulaw', action='store_true')
    parser.add_argument('--no-mulaw', dest='mulaw', action='store_false')
    parser.set_defaults(mulaw=True)
    parser.add_argument('--gpu', dest='is_gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='is_gpu', action='store_false')
    parser.set_defaults(is_gpu=False)
    parser.add_argument('--batch-sz', type=int, default=100,
                        help='size of the batch')
    args = parser.parse_args()

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # parse the metadata
    metadata = parse_metadata(fns)

    # load the model
    if args.model_path != 'mfcc':
        checkpoint = torch.load(args.model_path, lambda a, b: a)
        model = TASK_MODEL_MAP[args.task]()
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = args.model_path

    if args.is_gpu:
        model = model.cuda()

    # process!
    ext_latents(metadata.fn.values, args.out_fn, model,
                n_jobs=args.n_jobs, is_mulaw=args.mulaw,
                batch_sz=args.batch_sz)

    # save metadata
    metadata.to_csv(
        join(
            dirname(args.out_fn),
            basename(args.out_fn).split('.')[0] + '.csv'
        )
    )

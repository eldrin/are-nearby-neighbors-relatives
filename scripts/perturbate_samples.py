import os
from os.path import join, abspath, basename, splitext, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '..'))
sys.path.append(join(dirname(__file__), '..', 'audio-perturbator'))
sys.path.append(join(dirname(__file__), '..', 'audio-distances'))

import argparse
from functools import partial
import numpy as np
import librosa

from musiclatentconsistency.utils import save_mulaw, load_mulaw
from musiclatentconsistency.loudness import lufs_norm
from musiclatentconsistency.config import Config as cfg
from audiodistances.utils import parmap
from audioperturbator.transform import (PitchShifter,
                                        TimeStretcher,
                                        PinkNoiseMixer,
                                        PubAmbientMixer,
                                        MP3Compressor,
                                        Identity)

ALL_TRANSFORMERS = (PitchShifter, TimeStretcher, PinkNoiseMixer,
                    PubAmbientMixer, MP3Compressor)


def get_transform_range(transformer, perturbations):
    """Get transformation range according to the input transformer

    Args:
        transformer (BaseTransformer): instance of the transformer
        perturbations (OrderedDict): perturbation set
    
    Returns:
        list: a range of magnitudes w.r.t the perturbation
    """
    if isinstance(transformer, PitchShifter):
        return perturbations['PS']
    elif isinstance(transformer, TimeStretcher):
        return perturbations['TS']
    elif isinstance(transformer, PinkNoiseMixer):
        return perturbations['PN'] 
    elif isinstance(transformer, PubAmbientMixer):
        return perturbations['EN'] 
    elif isinstance(transformer, MP3Compressor):
        return perturbations['MP']  
    elif isinstance(transformer, Identity):
        return [0]
    else:
        raise NotImplementedError()


def get_suffix(transformer):
    """Get suffix according to the input transformer

    Args:
        transformer (BaseTransformer): instance of the transformer
    """
    if isinstance(transformer, PitchShifter):
        return 'PS'
    elif isinstance(transformer, TimeStretcher):
        return 'TS'
    elif isinstance(transformer, PinkNoiseMixer):
        return 'PN'
    elif isinstance(transformer, PubAmbientMixer):
        return 'EN'
    elif isinstance(transformer, MP3Compressor):
        return 'MP'
    elif isinstance(transformer, Identity):
        return 'OG'
    else:
        raise NotImplementedError()


def get_pert_id(transformer, magnitude):
    """Get id of given perturbation and mangitude

    Args:
        transformer (BaseTransformer): transformation class
        magnitude (int, float): level of perturbation

    Returns:
        str: identifier string
    """
    if isinstance(magnitude, int):
        pert_id_tmp = '{}_[{:d}]'
    elif isinstance(magnitude, float):
        pert_id_tmp = '{}_[{:.3f}]'
    else:
        pert_id_tmp = '{}_[{}]'

    pert_id = pert_id_tmp.format(
        get_suffix(transformer), magnitude
    )
    return pert_id


def _transform(fn, transformer, out_root,
               perturbations=cfg.PERTURBATIONS,sr=22050):
    """Transform given signal and save

    Args:
        fn (str): path to the input signal
        transformer (BaseTransformer): transformation class
        out_root (str): path to dump outputs
        perturbations (OrderedDict): perturbations set
        sr (int): sampling rate of given input signal
    """
    # load the signal
    if basename(fn).split('.')[-1] == 'npy':
        x = load_mulaw(fn)
    else:
        x, sr = librosa.load(fn, sr=sr)

    # transform
    for a in get_transform_range(transformer, perturbations):
        y = transformer(x, a)
        
        # normalization
        y = lufs_norm(y, x, cfg.FS)

        out_fn = '_'.join([
            basename(fn).split('.')[0],
            get_pert_id(transformer, a)
        ])

        # librosa.output.write_wav(join(out_root, out_fn), y, sr, norm=True)
        # save_mulaw(join(out_root, out_fn),
        #            librosa.util.normalize(y))
        save_mulaw(join(out_root, out_fn), y)


def transform(fns, out_root, perturbations=cfg.PERTURBATIONS,
              n_jobs=1):
    """Transform given audio files

    Args:
        fns (str): file name of the music
        out_root (str): path to dump files
        n_jobs (int): number of parallel jobs
    """
    for T in ALL_TRANSFORMERS:
        parmap(
            partial(_transform,
                    transformer=T(),
                    perturbations=perturbations,
                    out_root=out_root),
            fns, total=len(fns), n_workers=n_jobs, verbose=True
        )


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_files",
                        help='text file contains all the file names of music')
    parser.add_argument("out_path", help='path to dump output files')
    parser.add_argument("resolution_type", type=str, default='low',
                        help='resolution of perturbation {"low", "high", "all"}')
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args()

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]
        
    # set resolution
    if args.resolution_type == 'high':
        perturb = cfg.PERTURBATIONS
    elif args.resolution_type == 'low':
        perturb = cfg.PERTURBATIONS_HI_RES
    elif args.resolution_type == 'all':
        perturb = cfg.PERTURBATIONS_HI_LOW

    # process!
    transform(fns, args.out_path, perturb, n_jobs=args.n_jobs)

import os
from os.path import join, abspath, basename, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-perturbator'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-distances'))

import argparse
from functools import partial
import numpy as np
import librosa

from musiclatentconsistency.utils import save_mulaw, load_mulaw
from audiodistances.utils import parmap
from audioperturbator.transform import (PitchShifter,
                                        TimeStretcher,
                                        PinkNoiseMixer,
                                        PubAmbientMixer,
                                        MP3Compressor)

ALL_TRANSFORMERS = (PitchShifter, TimeStretcher, PinkNoiseMixer,
                    PubAmbientMixer, MP3Compressor)


def get_transform_range(transformer):
    """Get transformation range according to the input transformer

    Args:
        transformer (BaseTransformer): instance of the transformer
    """
    if isinstance(transformer, PitchShifter):
        return [-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12]  # 12
    elif isinstance(transformer, TimeStretcher):
        return [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]  # 12
    elif isinstance(transformer, PinkNoiseMixer):
        return [-15, -10, -6, -3, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]  # 14
    elif isinstance(transformer, PubAmbientMixer):
        return [-15, -10, -6, -3, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]  # 14
    elif isinstance(transformer, MP3Compressor):
        return [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192]  # 13
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
        pert_id_tmp = '{}_[{:.1f}]'
    else:
        pert_id_tmp = '{}_[{}]'

    pert_id = pert_id_tmp.format(
        get_suffix(transformer), magnitude
    )
    return pert_id


def _transform(fn, transformer, out_root, sr=22050):
    """Transform given signal and save

    Args:
        fn (str): path to the input signal
        transformer (BaseTransformer): transformation class
        out_root (str): path to dump outputs
        sr (int): sampling rate of given input signal
    """
    # load the signal
    if basename(fn).split('.')[-1] == 'npy':
        x = load_mulaw(fn)
    else:
        x, sr = librosa.load(fn, sr=sr)

    # transform
    for a in get_transform_range(transformer):
        y = transformer(x, a)

        out_fn = '_'.join(
            basename(fn).split('.')[0],
            get_pert_id(transformer, a)
        )

        # librosa.output.write_wav(join(out_root, out_fn), y, sr, norm=True)
        save_mulaw(join(out_root, out_fn), librosa.util.normalize(y))


def transform(fns, out_root, n_jobs=1):
    """Extract MFCCs

    Args:
        fns (str): file name of the music
        out_root (str): path to dump files
        n_mfcc (int): number of coefficients
        n_jobs (int): number of parallel jobs
    """
    for T in ALL_TRANSFORMERS:
        parmap(
            partial(_transform,
                    transformer=T(),
                    out_root=out_root),
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
    transform(fns, args.out_path, n_jobs=args.n_jobs)

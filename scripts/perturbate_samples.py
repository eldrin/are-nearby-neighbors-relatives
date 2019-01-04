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
        # return [-12, 12]
        return [-6, 6]
    elif isinstance(transformer, TimeStretcher):
        # return [0.25, 1.5]
        return [0.5, 1.25]
    elif isinstance(transformer, PinkNoiseMixer):
        # return [-15, 30]
        return [-9, 15]
    elif isinstance(transformer, PubAmbientMixer):
        # return [-15, 30]
        return [-9, 15]
    elif isinstance(transformer, MP3Compressor):
        # return [8, 192]
        return [64, 128]
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
        
        if isinstance(a, int):
            out_fn_tmp = '{}_{}_[{:d}].npy'
        elif isinstance(a, float):
            out_fn_tmp = '{}_{}_[{:f}].npy'
        else:
            out_fn_tmp = '{}_{}_[{}].npy'

        out_fn = out_fn_tmp.format(
            basename(fn).split('.')[0],
            get_suffix(transformer), a
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

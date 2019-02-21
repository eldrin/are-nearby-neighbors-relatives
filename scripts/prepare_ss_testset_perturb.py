import os
from os.path import join, dirname, basename
import sys
sys.path.append(join(dirname(__file__), '..'))
sys.path.append(join(dirname(__file__), '..', 'audio-perturbator'))
sys.path.append(join(dirname(__file__), '..', 'audio-distances'))

import argparse
from functools import partial
from tempfile import NamedTemporaryFile as NamedTmpF
import subprocess

import librosa
import numpy as np
import musdb
from tqdm import tqdm

from musiclatentconsistency.config import Config as cfg
from audiodistances.utils import parmap
from audioperturbator.transform import (PitchShifter,
                                        TimeStretcher,
                                        PinkNoiseMixer,
                                        PubAmbientMixer,
                                        MP3Compressor,
                                        Identity)

from perturbate_samples import _transform, get_pert_id, get_transform_range
from crop_audio import _random_crop_signal


ALL_TRANSFORMERS = (PitchShifter, TimeStretcher, PinkNoiseMixer,
                    PubAmbientMixer, MP3Compressor, Identity)


def save_mp3(path, y, sr):
    """Save given signal to mp3 file
    
    Args:
        path (str): path to save file
        y (np.ndarray): signal
        sr (int): sampling rate
    """
    with open(os.devnull, 'w') as shutup, NamedTmpF(suffix='.wav') as tmpf:
        
        librosa.output.write_wav(tmpf.name, y, sr)
        subprocess.call(
            ['ffmpeg', '-i', tmpf.name, '-codec:a',
             'libmp3lame', '-qscale:a', '2', path],
            stdout=shutup, stderr=shutup
        )
        
    
def _transform_and_save(out_root, track_name, signal,
                        sample_rate, start, end, transformer,
                        magnitude, signal_category, no_transform=False): 
    """"""
    if no_transform:
        transformed = signal[start:end]
    else:
        transformed = transformer(signal[start:end], magnitude)

    # get filename
    fn_tmp = '{}_{}_{:d}_{:d}_{}.mp3'.format(
        track_name, signal_category,
        start, end, get_pert_id(transformer, magnitude)
    )

    # save all of them
    save_mp3(
        join(out_root, fn_tmp),
        transformed, sample_rate
    )


def process(musdb_root, out_root, perturbations=cfg.PERTURBATIONS,
            n_samples=10, sig_len=44100 * 2): 
    """Crop multiple clips from a STEM, perturb and save
    
    Args:
        musdb_root (str): root of MUSDB18 STEM files
        out_root (str): path to dump files
        perturbations (OrderedDict): perturbations and their ranges
        n_samples (int): number of clips cropped
        sig_len (int): length to crop
    """
    # initiate the musdb module
    mus = musdb.DB(root_dir=musdb_root)

    # load the training tracks
    tracks = mus.load_mus_tracks(subsets=['test'])
    for track in tqdm(tracks, ncols=80):
        
        # get the audio mixture as numpy array shape=(num_sample, 2)
        mixture = track.audio.mean(-1)
        vocals = track.targets['vocals'].audio.mean(-1)
        accomp = track.targets['accompaniment'].audio.mean(-1)
        signals = [mixture, vocals, accomp]
        names = ['mixture', 'vocals', 'accomp']
        
        # get the sample rate
        sr = track.rate
        
        # get the slicing points
        starting_points = [
            x[1] for x
            in _random_crop_signal(
                mixture, n_samples, sig_len
            )
        ]
         
        # perturb each samples and save
        for T in ALL_TRANSFORMERS:
            transformer = T()
            
            for a in get_transform_range(transformer, perturbations):
                
                # crop the signals
                for start in starting_points:
                    end = start + sig_len 
                    for signal, name in zip(signals, names):
                        if (not isinstance(transformer,
                                           (PitchShifter, TimeStretcher)) and
                                ((name == 'vocals') or (name == 'accomp'))):
                            no_transform = True
                        else:
                            no_transform = False
                            
                        try:
                            _transform_and_save(
                                out_root, track.name, signal,
                                sr, start, end, transformer, a, name,
                                no_transform
                            )
                        except Exception as e:
                            print(e)


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("musdb_root",
                        help='text file contains all the file names of music')
    parser.add_argument("out_path", help='out path to save perturbed music files')
    parser.add_argument("resolution", default='low',
                        help='target resolution of pre-processing {"low", "high", "all"}') 
    args = parser.parse_args() 
    
    if args.resolution == 'hi':
        perturbations = cfg.PERTURBATIONS_HI_RES
    elif args.resolution == 'low':
        perturbations = cfg.PERTURBATIONS
    elif args.resolution == 'all':
        perturbations = cfg.PERTURBATIONS_HI_LOW
    else:
        raise ValueError(
            '[ERROR] given resolution "{}" is not supported!'
            .format(args.resolution)
        )

    # process!
    process(args.musdb_root, args.out_path, perturbations)
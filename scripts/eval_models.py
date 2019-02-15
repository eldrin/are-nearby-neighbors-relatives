import os
from os.path import join, abspath, basename, dirname, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'music-nn-models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-distances'))

from collections import namedtuple
import argparse
import re
from functools import partial
import pickle as pkl

import torch
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import accuracy_score
import museval
from tqdm import tqdm

from musiclatentconsistency.utils import (save_mulaw,
                                          load_mulaw,
                                          parse_metadata,
                                          parse_bracket)
from audiodistances.utils import parmap 
from musicnn.models import (VGGlike2DAutoTagger,
                            VGGlike2DAutoEncoder,
                            VGGlike2DUNet)
from musicnn.config import Config as cfg
from musicnn.datasets.autotagging import TAGS
from musicnn.datasets.instrecognition import CLS
from musicnn.datasets import files
from musicnn.evaluation.metrics import ndcg, apk, roc_auc_score


TASK_MODEL_MAP = {
    'auto_tagging': partial(VGGlike2DAutoTagger,
                            n_outputs=len(TAGS), layer1_channels=16),
    'inst_recognition': partial(VGGlike2DAutoTagger,
                                n_outputs=len(CLS), layer1_channels=16),
    'auto_encoder': partial(VGGlike2DAutoEncoder, layer1_channels=16),
    'source_separation': partial(VGGlike2DUNet, layer1_channels=16)
}
TASK_LABEL_MAP = {
    'auto_tagging': pkl.load(open(files.msd_lastfm50_label(), 'rb')),
    'inst_recognition': lambda audio_id: parse_bracket(audio_id)[0],
    'auto_encoder': None,
    'source_separation': None
}


def parse_fn(fn):
    """"""
    parsed = basename(fn).split('_')
    audio_id = '_'.join(parsed[:-4])
    return {
        'audio_id': audio_id,
        'start': parsed[-4],
        'end': parsed[-3],
        'transform': parsed[-2],
        'magnitude': parsed[-1]
    }


def _forward(fn, model, sr=22050):
    """Helper function to extract latent embedding for a single music file

    Args:
        fn (str): filename for a single music file
        out_root (str): path to dump output MFCCs
        sr (int): sampling rate
        
    Returns:
        np.ndarray: inference
    """
    if basename(fn).split('.')[-1] == 'npy':
        y = load_mulaw(fn)
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
    
    inp = torch.from_numpy(y)[None]
    return y, model(inp).data.numpy()[0]


def convert_idx_to_onehot(indices, n_labels):
    """"""
    y = np.zeros((n_labels,), dtype=int)
    y[indices] = 1
    return y


# for evaluation of bss
Track = namedtuple('Track', ['targets', 'rate'])
Target = namedtuple('Target', ['audio'])


def evaluate_clips(fns, model, task, batch_sz=128, verbose=False):
    """Evaluate given audio clips wrt tasks
    
    Args:
        fns (list of str): list contains filenames to evaluate
        model (VGGlike2D model): a model to test with clips
        task (str): corresponding task for model
                    {'auto_tagging', 'inst_recognition',
                     'auto_encoder', 'source_separation'}
        batch_sz (int): size of batch to process every iteration
        verbose (bool): verbosity flag
    
    Returns:
        pd.DataFrame: a table contains results
    """
    file_ext = '.' + fns[0].split('.')[-1]
    if verbose: fns = tqdm(fns, ncols=80) 
    
    # # get metadata
    # metadata = pd.DataFrame(
    #     [parse_fn(fn.replace(file_ext, '')) for fn in fns]
    # )
    # metadata['fns'] = fns
    
    TRUES, PREDS = {}, {}
    for fn in fns:
    # for fns_ in metadata.groupby(['transform', 'magnitude'])['fns'].apply(list):
        
        info = parse_fn(fn)
        key = '{}_{}'.format(
            info['transform'],
            info['magnitude'].split(file_ext)[0]
        )
        
        # register if not stored ever
        if key not in TRUES:
            TRUES[key] = []
        
        if key not in PREDS:
            PREDS[key] = []
        
        if task == 'source_separation':
            if not bool(np.random.binomial(1, 0.1)):
                continue
                
        # prepare data & forward
        inp, pred = _forward(fn, model)
        
        # retrieve the ground truth and measure clip-wise metric
        if task == 'auto_tagging':
            # retrieve tags
            true = [
                TAGS[tag] for tag
                in TASK_LABEL_MAP[task][info['audio_id'] + '.npy']
            ]
            TRUES[key].append(convert_idx_to_onehot(true, len(TAGS)))
            PREDS[key].append(pred)
            
        elif task == 'inst_recognition':
            # retrieve pre-dominant instrument
            true = CLS[TASK_LABEL_MAP[task](info['audio_id'])] 
            
            TRUES[key].append(true)
            PREDS[key].append(np.argmax(pred))
            
        elif task == 'auto_encoder':
            true, pred = pred 
            
            TRUES[key].append(true)
            PREDS[key].append(pred)
            
        elif task == 'source_separation':
            if '_mixture_' not in fn:
                continue
                
            phase = np.angle(
                librosa.stft(inp, n_fft=cfg.N_FFT, hop_length=cfg.HOP_SZ)
            )[None]
            pred_v = model._post_process(torch.from_numpy(pred)[None], phase)[0]
            pred_a = inp[:len(pred_v)] - pred_v
            
            # get ground truth sources
            true_v, _ = librosa.load(
                fn.replace('_mixture_', '_vocals_'), sr=cfg.SAMPLE_RATE)
            true_a, _ = librosa.load(
                fn.replace('_mixture_', '_accomp_'), sr=cfg.SAMPLE_RATE)
            
            TRUES[key].append((true_v[:len(pred_v), None], true_a[:len(pred_a), None]))
            PREDS[key].append((pred_v[:, None], pred_a[:, None])) 
        else:
            raise ValueError('[ERROR] Source separation is not supported yet!')      
    
    # calc metrics
    errors = []
    for transform in TRUES.keys():
        transform_name = transform.split('_')[0]
        transform_mag = float(parse_bracket(transform.split('_')[1])[0])
        
        if task == 'auto_tagging':
            t, p = np.array(TRUES[transform]), np.array(PREDS[transform])
            if not np.all(t.sum(axis=0) != 0):
                mask = t.sum(axis=0) != 0
                t, p = t[:, mask], p[:, mask]
            
            errors.append({
                'transform': transform_name,
                'magnitude': transform_mag,
                'ndcg@10': ndcg(t, p, k=10),
                'ap@10': apk(t, p, k=10),
                'roc-auc-track': roc_auc_score(t, p, average='samples'),
                'roc-auc-tag': roc_auc_score(t, p, average='macro')
            })
            
        elif task == 'inst_recognition':
            t, p = np.array(TRUES[transform]), np.array(PREDS[transform])
            errors.append({
                'transform': transform_name,
                'magnitude': transform_mag,
                'accuracy': accuracy_score(t, p)
            })

        elif task == 'auto_encoder':
            t, p = np.array(TRUES[transform]), np.array(PREDS[transform])
            errors.append({
                'transform': transform_name,
                'magnitude': transform_mag,
                'mse': np.mean((t - p)**2)
            })

        elif task == 'source_separation':
            result = {
                'transform': transform_name,
                'magnitude': transform_mag
            }
            
            local_results = []
            n_skipped = 0
            for (t_v, t_a), (p_v, p_a) in zip(TRUES[transform],
                                              PREDS[transform]):               
                if np.all(t_v == 0) or np.all(t_a == 0):
                    n_skipped += 1
                    continue
                    
                # make the length same
                shortest_v = min([t_v.shape[0], p_v.shape[0]])
                shortest_a = min([t_a.shape[0], p_a.shape[0]])
                t_v, p_v = t_v[:shortest_v], p_v[:shortest_v]
                t_a, p_a = t_a[:shortest_a], p_a[:shortest_a]
                    
                track = Track(
                    targets={
                        'vocals': Target(audio=t_v),
                        'accompaniment': Target(audio=p_v)
                    },
                    rate=cfg.SAMPLE_RATE
                ) 
                 
                local_results.append(
                    museval.eval_mus_track(
                        track,
                        {'vocals': p_v, 'accompaniment': p_a}
                    ).scores['targets']
                )
            # print('Total {:d} cases skipped!'.format(n_skipped))
            
            res = []
            for k, track in enumerate(local_results):
                for target in track:
                    for i, frame in enumerate(target['frames']):
                        res.append({ 
                            'transform': transform_name,
                            'magnitude': transform_mag,
                            'track': k, 
                            'target': target['name'],
                            'frame': i,
                        })
                        res[-1].update(frame['metrics'])
            errors.extend(res)
            
        else: 
            raise ValueError('[ERROR] Source separation is not supported yet!')        

    return pd.DataFrame(errors)


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("music_files",
                        help='text file contains all the file names of music')
    parser.add_argument("task", type=str, default='autotagging',
                        choices=set(TASK_MODEL_MAP.keys()),
                        help="type of the task of which the model is trained")
    parser.add_argument("model_path", help='path to model checkpoint dump')
    parser.add_argument("out_fn", help='filename to dump latent points and metadata')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]

    # load the model
    checkpoint = torch.load(args.model_path, lambda a, b: a)
    model = TASK_MODEL_MAP[args.task]()
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    
    # process!
    results = evaluate_clips(fns, model, args.task, verbose=True)
    
    # save
    results.to_csv(args.out_fn)
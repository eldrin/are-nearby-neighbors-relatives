import os
from os.path import join, abspath, basename, dirname, splitext
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'music-nn-models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-distances'))

import argparse
import re
from functools import partial
import torch
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import accuracy_score

from musiclatentconsistency.utils import save_mulaw, load_mulaw, parse_metadata
from audiodistances.utils import parmap
from musicnn.models import (VGGlike2DAutoTagger,
                            VGGlike2DAutoEncoder,
                            VGGlike2DUNet)
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
    'inst_recognition': lambda audio_id: re.findall('\[(.*?)\]', audio_id)[0],
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

    return model(torch.from_numpy(y)[None]).data.numpy()[0]


def convert_idx_to_onehot(indices, n_labels):
    """"""
    y = np.zeros((n_labels,), dtype=int)
    y[indices] = 1
    return y


def evaluate_clips(fns, model, task, verbose=False):
    """Evaluate given audio clips wrt tasks
    
    Args:
        fns (list of str): list contains filenames to evaluate
        model (VGGlike2D model): a model to test with clips
        task (str): corresponding task for model
                    {'auto_tagging', 'inst_recognition',
                     'auto_encoder', 'source_separation'}
    
    Returns:
        pd.DataFrame: a table contains results
    """
    if verbose: fns = tqdm(fns, ncols=80)
        
    TRUES, PREDS = {}, {}
    for fn in fns:
        info = parse_fn(fn)
        key = '{}_{}'.format(info['transform'], info['magnitude'])
        
        # register if not stored ever
        if key not in TRUES:
            TRUES[key] = []
        
        if key not in PREDS:
            PREDS[key] = []
        
        # prepare data & forward
        pred = _forward(fn, model) 
        
        # retrieve the ground truth and measure clip-wise metric
        if task == 'auto_tagging':
            # retrieve tags
            true = [
                TAGS[tag] for tag
                in TASK_LABEL_MAP[task][info['audio_id'] + '.npy']:
            ]
            TRUES[key].append(convert_idx_to_onehot(true, len(TAGS)))
            PREDS[key].append(pred)
            
        elif task == 'inst_recognition':
            # retrieve pre-dominant instrument
            true = CLS[TASK_LABEL_MAP[task](info['audio_id'])] 
            true = convert_idx_to_onehot(true, len(CLS))
            
            TRUES[key].append(true)
            PREDS[key].append(np.argmax(pred))
            
        elif task == 'auto_encoder':
            true, pred = pred 
            
            TRUES[key].append(true)
            PREDS[key].append(pred)
            
        elif task == 'source_separation':
            raise ValueError('[ERROR] Source separation is not supported yet!')
        else: 
            raise ValueError('[ERROR] Source separation is not supported yet!')        
    
    # calc metrics
    errors = []
    if task == 'auto_tagging':
        TRUES, PREDS = np.array(TRUES), np.array(PREDS)
        errors = {
            'ndcg@10': ndcg(TRUES, PREDS, k=10),
            'ap@10': apk(TRUES, PREDS, k=10),
            'roc-auc-track': roc_auc_score(TRUES, PREDS, average='samples'),
            'roc-auc-tag': roc_auc_score(TRUES, PREDS, average='macro')
        }
    elif task == 'inst_recognition':
        errors = {
            'accuracy': accuracy_score(TRUES, PREDS)
        }
        
    elif task == 'auto_encoder':
        TRUES, PREDS = np.array(TRUES), np.array(PREDS)
        errors = {
            'mse': np.mean((TRUES - PREDS)**2)
        }
        
    elif task == 'source_separation':
        raise ValueError('[ERROR] Source separation is not supported yet!')
    else: 
        raise ValueError('[ERROR] Source separation is not supported yet!')        
            
    return errors


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
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args() 

    # load the file list
    with open(args.music_files) as f:
        fns = [l.replace('\n', '') for l in f.readlines()]
    
    # parse the metadata
    metadata = parse_metadata(fns)

    # load the model
    checkpoint = torch.load(args.model_path, lambda a, b: a)
    model = TASK_MODEL_MAP[args.task]()
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    
    # process!
    results = evaluate_clips(metadata.fn.values, args.out_fn, model, n_jobs=args.n_jobs)
    
    # save metadata
    metadata.to_csv(
        join(
            dirname(args.out_fn),
            basename(args.out_fn).split('.')[0] + '.csv'
        )
    )
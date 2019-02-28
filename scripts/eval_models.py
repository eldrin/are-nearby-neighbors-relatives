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
from sklearn.metrics import accuracy_score, r2_score
import museval
from tqdm import tqdm

from musiclatentconsistency.utils import (save_mulaw,
                                          load_mulaw,
                                          parse_metadata,
                                          parse_bracket,
                                          parse_fn,
                                          pad_or_crop)
from audiodistances.utils import parmap 
from musicnn.models import (VGGlike2DAutoTagger,
                            VGGlike2DAutoEncoder,
                            VGGlike2DUNet,
                            MFCCAutoTagger,
                            MFCCAutoEncoder,
                            MFCCAESourceSeparator)
from musicnn.config import Config as cfg
from musicnn.datasets.autotagging import TAGS
from musicnn.datasets.instrecognition import CLS
from musicnn.datasets import files
from musicnn.evaluation.metrics import _ndcg, _apk, roc_auc_score

from ext_latent import _mfcc

AT_LABEL_MAP = pkl.load(open(files.msd_lastfm50_label(), 'rb'))
TASK_MODEL_MAP = {
    'auto_tagging': partial(VGGlike2DAutoTagger,
                            n_outputs=len(TAGS), layer1_channels=16),
    'auto_tagging_mfcc': partial(MFCCAutoTagger, n_outputs=len(TAGS)),
    'inst_recognition': partial(VGGlike2DAutoTagger,
                                n_outputs=len(CLS), layer1_channels=16),
    'inst_recognition_mfcc': partial(MFCCAutoTagger, n_outputs=len(CLS)),
    'auto_encoder': partial(VGGlike2DAutoEncoder, layer1_channels=16),
    'auto_encoder_mfcc': MFCCAutoEncoder,
    'source_separation': partial(VGGlike2DUNet, layer1_channels=16),
    'source_separation_mfcc': MFCCAESourceSeparator,
}
TASK_LABEL_MAP = {
    'auto_tagging': AT_LABEL_MAP,
    'inst_recognition': lambda audio_id: parse_bracket(audio_id)[0],
    'auto_encoder': None,
    'source_separation': None
}
PERTURBATIONS = {'TS', 'PS', 'PN', 'EN', 'MP'}


def load_audio(fn):
    """""" 
    if basename(fn).split('.')[-1] == 'npy':
        y = load_mulaw(fn)
    else:
        y, sr = librosa.load(fn, sr=sr)
    
    # make sure the input is right dtype
    y = y.astype(np.float32)
    
    return y


def _forward(y, model, sr=22050):
    """Helper function to extract latent embedding for a single music file

    Args:
        y (np.ndarray): input signal 
        out_root (str): path to dump output MFCCs
        sr (int): sampling rate
        
    Returns:
        np.ndarray: inference
    """
    inp = torch.from_numpy(y)[None]
    infer = model(inp)
    if isinstance(model, (VGGlike2DAutoEncoder, MFCCAutoEncoder)):
        infer = (infer[0].data.numpy()[0], infer[1].data.numpy()[0])
    else:
        infer = infer.data.numpy()[0]
        
    return y, infer


def convert_idx_to_onehot(indices, n_labels):
    """"""
    y = np.zeros((n_labels,), dtype=int)
    y[indices] = 1
    return y


# for evaluation of bss
Track = namedtuple('Track', ['targets', 'rate'])
Target = namedtuple('Target', ['audio'])


def evaluate_clips(fns, model, task, batch_sz=128, normalize=False, verbose=False):
    """Evaluate given audio clips wrt tasks
    
    Args:
        fns (list of str): list contains filenames to evaluate
        model (VGGlike2D model): a model to test with clips
        task (str): corresponding task for model
                    {'auto_tagging', 'inst_recognition',
                     'auto_encoder', 'source_separation'}
        batch_sz (int): size of batch to process every iteration
        normalize (bool): normalization setup
        verbose (bool): verbosity flag
    
    Returns:
        pd.DataFrame: a table contains results
    """
    metadata = pd.DataFrame([parse_fn(fn) for fn in fns])
    metadata['fn'] = fns
    
    # # sampling 100 original for sanity check
    # og100 = metadata[metadata['transform'] == 'OG']['audio_id'].sample(100)
    # metadata = metadata[metadata['audio_id'].isin(set(og100))] 
    
    # calculate all the original files mean dB
    if normalize:
        mean_dbs = []
        for fn in tqdm(metadata['fn'].values, ncols=80):
            y = load_audio(fn)
            mean_dbs.append(librosa.amplitude_to_db(abs(librosa.stft(y))).mean())
        metadata['meandB'] = mean_dbs
    
    file_ext = '.' + fns[0].split('.')[-1]
    if verbose: fns_ = tqdm(metadata['fn'].values, ncols=80)
    else: fns_ = metadata['fn'].values
    
    TRUES, PREDS = {}, {}  # for global metrics (ROC-AUC)
    errors = []
    for fn in fns_:
        info = parse_fn(fn)
        transform_name = info['transform']
        transform_mag = float(
            parse_bracket(info['magnitude'].split(file_ext)[0])[0]
        )
        key = (transform_name, transform_mag)
        
        if key not in TRUES:
            TRUES[key] = []
            
        if key not in PREDS:
            PREDS[key] = []
        
        if ('source_separation' in task) and ('_mixture_' not in fn):
            continue
                
        # prepare data & forward
        # normalize the transformed data
        # find the original audio's mean dB
        if normalize:
            mean_db_og = metadata[
                (metadata['audio_id'] == info['audio_id']) &
                (metadata['transform'] == 'OG')
            ]['meandB'].values
            mean_db_cur = metadata[
                (metadata['audio_id'] == info['audio_id']) &
                (metadata['transform'] == info['transform']) & 
                (metadata['magnitude'] == info['magnitude'])
            ]['meandB'].values
            coef = np.float32(10**((mean_db_og - mean_db_cur) / 20))
        else:
            coef = np.float32(1)
        
        y = load_audio(fn) * coef
        y = pad_or_crop(y, model.sig_len)
        inp, pred = _forward(y, model)
        
        # retrieve the ground truth and measure clip-wise metric
        if 'auto_tagging' in task:
            k = 10
            
            # retrieve tags
            t = [
                TAGS[tag] for tag
                in TASK_LABEL_MAP['auto_tagging'][info['audio_id'] + '.npy']
            ]
            p = np.argsort(-pred)[:k]
            
            # register for global metrics
            TRUES[key].append(convert_idx_to_onehot(t, len(TAGS)))
            PREDS[key].append(pred)
            
            errors.append({
                'track': info['audio_id'],
                'transform': transform_name,
                'magnitude': transform_mag,
                'ndcg@10': _ndcg(t, p, k=k),
                'ap@10': _apk(t, p, k=k),
            })
            
        elif 'inst_recognition' in task:
            # retrieve pre-dominant instrument
            t = CLS[TASK_LABEL_MAP['inst_recognition'](info['audio_id'])] 
            p = np.argmax(pred)
            
            errors.append({
                'track': info['audio_id'],
                'transform': transform_name,
                'magnitude': transform_mag,
                'accuracy': 1 if t == p else 0,
            })
            
        elif 'auto_encoder' in task:
            true, pred = pred 
            
            errors.append({
                'track': info['audio_id'],
                'transform': transform_name,
                'magnitude': transform_mag,
                'mse': np.mean((true - pred)**2),
                'r2': r2_score(true.ravel(), pred.ravel())
            })
            
        elif 'source_separation' in task:
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
            # pre-process the length
            true_v = pad_or_crop(true_v, model.sig_len)
            true_a = pad_or_crop(true_a, model.sig_len)
             
            result = {
                'transform': transform_name,
                'magnitude': transform_mag
            }
            
            local_results = []
            n_skipped = 0

            # make the length same
            shortest_v = min([true_v.shape[0], pred_v.shape[0]])
            shortest_a = min([true_a.shape[0], pred_a.shape[0]])
            t_v, p_v = true_v[:shortest_v, None], pred_v[:shortest_v, None]
            t_a, p_a = true_a[:shortest_a, None], pred_a[:shortest_a, None]
            
            if np.all(t_v == 0) or np.all(t_a == 0):
                n_skipped += 1
                continue
                    
            # wrap given input track
            track = Track(
                targets={
                    'vocals': Target(audio=t_v),
                    'accompaniment': Target(audio=p_v)
                },
                rate=cfg.SAMPLE_RATE
            ) 
            
            # evaluate
            res = museval.eval_mus_track(
                track,
                {'vocals': p_v, 'accompaniment': p_a}
            ).scores['targets']
            # for normalization
            n_res = museval.eval_mus_track(
                track,
                {'vocals': inp[:, None], 'accompaniment': inp[:, None]}
            ).scores['targets']
            
            # for each target {'vocals', 'accompaniment'} the result
            for target, n_target in zip(res, n_res):
                for i, (frame, n_frame) in enumerate(zip(target['frames'], n_target['frames'])):
                    out = {
                        'track': info['audio_id'],
                        'transform': transform_name,
                        'magnitude': transform_mag,
                        'target': target['name'],
                        'frame': i
                    }
                    out.update({
                        key: frame['metrics'][key] - n_frame['metrics'][key]
                        for key in frame['metrics'].keys()
                    })
                    errors.append(out) 
        else:
            raise ValueError('[ERROR] Given task is not supported yet!')      
             
    # calc global metrics
    for transform in TRUES.keys():
        
        if 'auto_tagging' in task:
            t, p = np.array(TRUES[transform]), np.array(PREDS[transform])
            if not np.all(t.sum(axis=0) != 0):
                mask = t.sum(axis=0) != 0
                t, p = t[:, mask], p[:, mask]  
                
            errors.append({
                'transform': transform[0],
                'magnitude': transform[1],
                'roc-auc-track': roc_auc_score(t, p, average='samples'),
                'roc-auc-tag': roc_auc_score(t, p, average='macro')
            })
        elif any(
            [t in task for t
             in {'auto_encoder', 'source_separation', 'inst_recognition'}]):
            # no global metric is required to these cases
            continue
        else: 
            raise ValueError('[ERROR] Given task is not supported yet!')        

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
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=False)
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
    results = evaluate_clips(fns, model, args.task,
                             normalize=args.normalize, verbose=True)

    # save
    results.to_csv(args.out_fn)

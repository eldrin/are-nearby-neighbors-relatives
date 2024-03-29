import os
from os.path import join, dirname, basename
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio-perturbator'))

import random
import glob
import argparse

from tqdm import tqdm

from perturbate_samples import get_transform_range, get_suffix, get_pert_id
from audioperturbator.transform import (PitchShifter,
                                        TimeStretcher,
                                        PinkNoiseMixer,
                                        PubAmbientMixer,
                                        MP3Compressor)

TRANSFORMATION_ORDER = []
for transform in (PitchShifter, TimeStretcher, PinkNoiseMixer,
                  PubAmbientMixer, MP3Compressor):
    T = transform()
    for magnitude in get_transform_range(T):
        TRANSFORMATION_ORDER.append(get_pert_id(T, magnitude))

# TRANSFORMATION_ORDER = [
#     'PN_[30]', 'PN_[-15]',
#     'EN_[30]', 'EN_[-15]',
#     'PS_[-12]', 'PS_[12]',
#     'TS_[1.5]', 'TS_[0.2]',
#     'MP_[8]', 'MP_[192]'
# ]


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("original_path",
                        help='path to the original clips mfcc files')
    parser.add_argument("transformed_path",
                        help='path to transformed clips mfcc files')
    parser.add_argument("output_fn", help="output list text filename")
    parser.add_argument("--type", type=str, default="*.npy",
                        help="file extension of the target files")
    parser.add_argument("--n-items", type=int, default=1000,
                        help='number of target items')
    parser.add_argument("--n-jobs", type=int, help='number of parallel jobs')
    args = parser.parse_args()

    # load original fns
    org_fns = glob.glob(join(args.original_path, args.type))
    tns_fns = glob.glob(join(args.transformed_path, args.type))

    if args.n_items >= len(org_fns):
        n_items = len(org_fns)
    else:
        n_items = args.n_items
    random.shuffle(org_fns)

    with open(args.output_fn, 'w') as f:
        # for each original 
        tns_fns_new = []
        for fn in tqdm(org_fns[:n_items], ncols=80):
            fn_id = basename(fn).split('.')[0]

            # write original file's fn
            f.write(fn + '\n')

            # filter transformation for this item
            tns = [fn for fn in tns_fns if fn_id in fn]
            # re-order
            tns_ = []
            for tns_type in TRANSFORMATION_ORDER:
                tns_.append([fn for fn in tns if tns_type in fn][0])
            tns_fns_new.extend(tns_)

        for fn in tqdm(tns_fns_new, ncols=80):
            f.write(fn + '\n')

    print('N_ORIGINALS: {:d}'.format(n_items))

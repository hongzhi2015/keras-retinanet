#!/usr/bin/env python

import sys
import os
import argparse
import pickle

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..utils.plotx import plot_diag_summ, plot_diag_detail


def parse_args(args):
    ap = argparse.ArgumentParser(description='Plot P-R curve')
    ap.add_argument('--image-dir', help='where images are.', required=True)
    ap.add_argument('-d', '--desc', required=False, default=None, help='Description of the curve')
    ap.add_argument('-m', '--metrics', required=True, help='Path to pickled metrics')
    ap.add_argument('-o', '--output-dir', required=True, help='Output directory path')

    gp = ap.add_argument_group('Detail exposure options')
    gp.add_argument('--min-score', required=True, default=0.0, type=float, help='Only keep bboxes whose scores >= this value')

    return ap.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    with open(args.metrics, 'rb') as f:
        diag = pickle.load(f)

    plot_diag_summ(args.desc, diag, args.output_dir)
    plot_diag_detail(image_root=args.image_dir, diag=diag, min_score=args.min_score, out_dir=args.output_dir)


if __name__ == '__main__':
    main()

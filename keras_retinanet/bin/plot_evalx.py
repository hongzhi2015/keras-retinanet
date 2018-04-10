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
from ..utils.evalx import CookedDiagnostic
from ..utils.plotx import plot_diag_summ, plot_diag_detail


def parse_args(args):
    def add_score_threshold(ap):
        ap.add_argument('--score-threshold', help='Threshold on score to filter detections with.', default=0.05, type=float)

    def add_iou_threshold(ap):
        ap.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection.', default=0.5, type=float)

    def add_metrics_path(ap):
        ap.add_argument('-m', '--metrics', required=True, help='Path to pickled metrics')

    def add_out_dir_path(ap):
        ap.add_argument('-o', '--output-dir', required=True, help='Output directory path')

    ap = argparse.ArgumentParser(description='Plot evaluation-X result',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_parsers = ap.add_subparsers(dest='sub_cmd', help='Plot types')

    ###########
    # Summary #
    ###########
    sp = sub_parsers.add_parser('summary', help='Plot summary curves',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_metrics_path(sp)
    add_score_threshold(sp)
    add_iou_threshold(sp)
    sp.add_argument('-d', '--desc', required=False, default=None, help='Description of curves')
    add_out_dir_path(sp)

    ##########
    # Detail #
    ##########
    sp = sub_parsers.add_parser('detail', help='Plot detection details')
    add_metrics_path(sp)
    sp.add_argument('--image-dir', help='image root directory', required=True)
    add_score_threshold(sp)
    add_iou_threshold(sp)
    add_out_dir_path(sp)

    return ap.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    with open(args.metrics, 'rb') as f:
        raw_diag = pickle.load(f)

    if args.sub_cmd == 'summary':
        cooked_diag = CookedDiagnostic(raw_diag=raw_diag,
                                       iou_thresh=args.iou_threshold,
                                       score_range=(args.score_threshold, 1.0))

        plot_diag_summ(args.desc, cooked_diag, args.output_dir)
    elif args.sub_cmd == 'detail':
        # FIXME: FINISH THIS
        assert False, 'IMPLEMENT THIS'
        plot_diag_detail(image_root=args.image_dir, diag=diag, min_score=args.min_score, out_dir=args.output_dir)
    else:
        assert False, 'Never be here'


if __name__ == '__main__':
    main()

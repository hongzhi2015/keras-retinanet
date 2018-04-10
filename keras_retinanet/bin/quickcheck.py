#!/usr/bin/env python3

import sys
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"


from ..utils.eval import Diagnostic


def parse_args(args):
    ap = argparse.ArgumentParser(description='Plot Score accuracy > 0.9 curve')
    ap.add_argument('-m', '--metrics', required=True, help='Path to pickled metrics')
    return ap.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    with open(args.metrics, 'rb') as f:
        diag = pickle.load(f)

    scores = []
    ge_90_ratios = []
    for score in np.arange(0, 1, 0.05):
        scores.append(score)
        ge_90_ratios.append(get_ge_90_ratio(diag, score))

    plt.plot(scores, ge_90_ratios)
    plt.xlabel('Score')
    plt.ylabel('Ratio of images whose prediction precision >= 0.9')
    plt.show()


def get_img_precision(diag, score, img_path):
    lbl2det = diag.get_image_detection(img_path)
    new_dets = [d.ge_min_score(score) for d in lbl2det.values()]
    ann_cnt = 0
    tp_cnt = 0
    fp_cnt = 0

    for det in new_dets:
        ann_cnt += len(det.annotations)
        tp_cnt += len(det.true_positives)
        fp_cnt += len(det.false_positives)

    if tp_cnt + fp_cnt == 0:
        return 0.0
    else:
        return tp_cnt / (tp_cnt + fp_cnt)


def get_ge_90_ratio(diag, score):
    image_paths = list(diag.iter_image_paths())
    ge_90_cnt = 0

    for img_path in image_paths:
        prec = get_img_precision(diag, score, img_path)
        if prec >= 0.90:
            ge_90_cnt += 1

    return ge_90_cnt / len(image_paths)


main()

#!/usr/bin/env python

import sys
import os
import argparse
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"


def f_beta(beta, p, r):
    beta_square = beta ** 2
    return (1 + beta_square) * p * r / (beta_square * p + r)


def _plot_1_pr(ax, label, ap, recalls, precisions):
    """
    Plot PR curve to ax.
    """
    assert recalls.shape == precisions.shape
    ax.step(recalls, precisions, color='b', alpha=0.2, where='post')
    ax.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Label {0:} : AUC={1:0.2f}'.format(label, ap))


def _plot_pr(desc, all_average_precisions, all_precisions, all_recalls):
    """
    Return Figure of PR curves.
    """
    grid = int(math.ceil(math.sqrt(len(all_average_precisions))))
    fig = plt.figure('PR', figsize=(9, 10), dpi=100)
    axes = fig.subplots(nrows=grid, ncols=grid, squeeze=False)

    for label, (_, ax) in zip(all_average_precisions.keys(), np.ndenumerate(axes)):
        ap = all_average_precisions[label]
        print(label, '{:.4f}'.format(ap))
        _plot_1_pr(ax, label, ap, all_recalls[label], all_precisions[label])

    mAP_str = "Mean average precision : {:0.2f}".format(sum(all_average_precisions.values()) / len(all_average_precisions))
    sup_title = '\n'.join([desc, mAP_str]) if desc else mAP_str
    fig.suptitle(sup_title)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def _plot_1_f1(ax, label, precisions, recalls, scores):
    assert precisions.shape == recalls.shape == scores.shape
    ax.plot(scores, precisions, '-', label='Precision')
    ax.plot(scores, recalls, '-', label='Recall')
    ax.plot(scores, f_beta(1.0, precisions, recalls), '--', label='F1')
    ax.plot(scores, f_beta(0.5, precisions, recalls), '--', label='F0.5')
    ax.plot(scores, f_beta(2.0, precisions, recalls), '--', label='F2')
    ax.grid()
    ax.legend()
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Score')
    ax.set_ylabel('Precision, Recall, FBeta')
    ax.set_title('Label {}'.format(label))
    pass


def _plot_f1(desc, all_precisions, all_recalls, all_scores):
    grid = int(math.ceil(math.sqrt(len(all_precisions))))
    fig = plt.figure('F1', figsize=(9, 10), dpi=100)
    axes = fig.subplots(nrows=grid, ncols=grid, squeeze=False)

    for label, (_, ax) in zip(all_precisions.keys(), np.ndenumerate(axes)):
        _plot_1_f1(ax, label, all_precisions[label], all_recalls[label], all_scores[label])

    if desc:
        fig.suptitle(desc)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def _do_plot(desc, metric_path, out_path):
    with open(metric_path, 'rb') as f:
        metrics = pickle.load(f)

    all_average_precisions = metrics['average_precisions']
    all_precisions = metrics['precisions']
    all_recalls = metrics['recalls']
    all_scores = metrics['scores']

    pr_fig = _plot_pr(desc, all_average_precisions, all_precisions, all_recalls)
    f1_fig = _plot_f1(desc, all_precisions, all_recalls, all_scores)

    root, ext = os.path.splitext(out_path)
    pr_fig.savefig(root + '.pr' + ext)
    f1_fig.savefig(root + '.f1' + ext)

    plt.show()
    pass


def parse_args(args):
    ap = argparse.ArgumentParser(description='Plot P-R curve')
    ap.add_argument('-d', '--desc', required=False, default=None, help='Description of the curve')
    ap.add_argument('-m', '--metrics', required=True, help='Path to pickled metrics')
    ap.add_argument('-o', '--output', required=True, help='Output image path')
    return ap.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    _do_plot(args.desc, args.metrics, args.output)


if __name__ == '__main__':
    main()

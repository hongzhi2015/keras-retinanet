import math
import numpy as np
import matplotlib.pyplot as plt
import os

from .evaluate import CookedDiagnostic, LabelDetection


def plot_summ(desc, cooked_diag, out_dir):
    """
    Plot summary of given Diagnostic instance.

    desc    str     description of the Diagnostic
    diag    Diagnostic instance
    out_dir str     output directory
    """
    assert isinstance(cooked_diag, CookedDiagnostic)

    pr_fig = _plot_pr(desc, cooked_diag)
    pr_score_fig = _plot_pr_score(desc, cooked_diag)

    os.makedirs(out_dir, exist_ok=True)
    pr_fig.savefig(os.path.join(out_dir, 'pr.jpg'))
    pr_score_fig.savefig(os.path.join(out_dir, 'pr_score.jpg'))
    plt.show()


def f_beta(beta, p, r):
    beta_square = beta ** 2
    return (1 + beta_square) * p * r / (beta_square * p + r)


def _plot_1_pr(ax, label, lbl_det):
    """
    Plot PR curve to ax.
    """
    assert isinstance(lbl_det, LabelDetection)
    ax.step(lbl_det.recalls, lbl_det.precisions, color='b', alpha=0.2, where='post')
    ax.fill_between(lbl_det.recalls, lbl_det.precisions, step='post', alpha=0.2, color='b')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Label {0:} : AUC={1:0.2f}'.format(label, lbl_det.average_precision))


def _plot_pr(desc, cooked_diag):
    """
    Return Figure of PR curves.
    """
    labels = cooked_diag.get_labels()
    grid = int(math.ceil(math.sqrt(len(labels))))
    fig = plt.figure('PR', figsize=(9, 10), dpi=100)
    axes = fig.subplots(nrows=grid, ncols=grid, squeeze=False)

    for label, (_, ax) in zip(labels, np.ndenumerate(axes)):
        lbl_det = cooked_diag.get_label_detection(label)
        print(label, '{:.4f}'.format(lbl_det.average_precision))
        _plot_1_pr(ax, label, lbl_det)

    mAP_str = "Mean average precision : {:0.2f}".format(cooked_diag.get_mAP())
    sup_title = '\n'.join([desc, mAP_str]) if desc else mAP_str
    fig.suptitle(sup_title)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def _plot_1_pr_score(ax, label, lbl_det):
    """
    Plot Precision, Recall / Score curves.
    """
    assert isinstance(lbl_det, LabelDetection)
    ax.plot(lbl_det.scores, lbl_det.precisions, '-', label='Precision')
    ax.plot(lbl_det.scores, lbl_det.recalls, '-', label='Recall')
    ax.plot(lbl_det.scores, f_beta(1.0, lbl_det.precisions, lbl_det.recalls), '--', label='F1')
    ax.plot(lbl_det.scores, f_beta(0.5, lbl_det.precisions, lbl_det.recalls), '--', label='F0.5')
    ax.plot(lbl_det.scores, f_beta(2.0, lbl_det.precisions, lbl_det.recalls), '--', label='F2')
    ax.grid()
    ax.legend()
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Score')
    ax.set_ylabel('Precision, Recall, FBeta')
    ax.set_title('Label {}'.format(label))
    pass


def _plot_pr_score(desc, cooked_diag):
    labels = cooked_diag.get_labels()
    grid = int(math.ceil(math.sqrt(len(labels))))
    fig = plt.figure('PR/Score', figsize=(9, 10), dpi=100)
    axes = fig.subplots(nrows=grid, ncols=grid, squeeze=False)

    for label, (_, ax) in zip(labels, np.ndenumerate(axes)):
        lbl_det = cooked_diag.get_label_detection(label)
        _plot_1_pr_score(ax, label, lbl_det)

    if desc:
        fig.suptitle(desc)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig

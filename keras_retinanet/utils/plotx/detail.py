import os
import math
import numpy as np
import cv2
from collections import namedtuple

from ..evalx import Diagnostic, ImageDetection


def plot_diag_detail(image_root, diag, min_score, out_dir):
    # Dump details
    details_dir = os.path.join(out_dir, 'details')
    os.makedirs(details_dir, exist_ok=True)

    dtl_img_path_lbl2det = _plot_details(image_root, diag, min_score, details_dir)
    dtl_img_path_negcnt_lst = _get_img_neg_cnts(dtl_img_path_lbl2det)

    # Dump false positives
    fp_dir = os.path.join(out_dir, 'falsepositives')
    os.makedirs(fp_dir, exist_ok=True)

    sorted_by_fp = sorted(dtl_img_path_negcnt_lst, key=lambda x: -x[1].fp_cnt)
    index_max = len(sorted_by_fp)
    index_len_max = int(math.floor(math.log10(index_max)) + 1)
    filename_fmt = '{{:0{}}}_fp_{{}}{{}}'.format(index_len_max)

    for i, (dtl_img_path, neg_cnt) in enumerate(sorted_by_fp, start=1):
        _, ext = os.path.splitext(dtl_img_path)
        ln_path = os.path.join(fp_dir, filename_fmt.format(i, neg_cnt.fp_cnt, ext))
        os.symlink(os.path.relpath(dtl_img_path, start=fp_dir), ln_path)


def _plot_details(image_root, diag, min_score, details_dir):
    """
    Return [(output detail image path,  {label: ImageDetection})]
    """
    ret = []
    for img_path in diag.iter_image_paths():
        lbl2det = diag.get_image_detection(img_path)
        # replace detection by min_score
        lbl2det = dict((lbl, det.ge_min_score(min_score))
                       for lbl, det in lbl2det.items())
        img_real_path = os.path.join(image_root, img_path)
        detail_img = _plot_detection(img_real_path, lbl2det)

        out_path = os.path.join(details_dir, img_path)
        out_dir = os.path.dirname(out_path)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        cv2.imwrite(out_path, detail_img)
        ret.append((out_path, lbl2det))

    return ret


def _plot_detection(img_real_path, lbl2det):
    """
    Return an image.

    image_path  path to the image file
    lbl2det     {label: ImageDetection}
    """
    GND_TRUTH_COLOR = (0, 255, 0)
    TRUE_POS_COLOR  = (199, 199, 0)
    FALSE_POS_COLOR = (0, 0, 255)
    FALSE_NEG_COLOR = (255, 0, 0)

    GND_TRUTH_THICKNESS = 5
    OTHER_THICKNESS = 2

    canvas = cv2.imread(img_real_path, cv2.IMREAD_COLOR)

    for det in lbl2det.values():
        # ground truth
        for ann in det.annotations:
            x0, y0, x1, y1 = ann
            cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color=GND_TRUTH_COLOR, thickness=GND_TRUTH_THICKNESS)

        # true positives
        for tp in det.true_positives:
            x0, y0, x1, y1 = tp[:4]
            cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color=TRUE_POS_COLOR, thickness=OTHER_THICKNESS)

        # false positives
        for fp in det.false_positives:
            x0, y0, x1, y1 = fp[:4]
            cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color=FALSE_POS_COLOR, thickness=OTHER_THICKNESS)

        return canvas


# Negative counts of an image
# fp_cnt        false positives count
# fn_cnt        false negative count
NegCnts = namedtuple('NegCnts', ['fp_cnt', 'fn_cnt'])


def _get_img_neg_cnts(detail_img_path_lbl2det):
    """
    Return [(detail image path, NegCnts)]

    img_path_lbl2det    [(detail image path, {label: ImageDetection})]
    """
    ret = []

    for detail_img_path, lbl2det in detail_img_path_lbl2det:
        fp_cnt = 0
        fn_cnt = 0

        for img_det in lbl2det.values():
            assert isinstance(img_det, ImageDetection)
            fp_cnt += len(img_det.false_positives)
            # FIXME: No false negative yet

        ret.append((detail_img_path, NegCnts(fp_cnt=fp_cnt, fn_cnt=fn_cnt)))

    return ret

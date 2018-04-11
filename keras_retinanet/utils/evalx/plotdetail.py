import os
import math
import cv2
from collections import namedtuple

from .evaluate import CookedDiagnostic, CookedDetection


def plot_detail(image_root, cooked_diag, out_dir):
    assert isinstance(cooked_diag, CookedDiagnostic)
    # Dump details
    details_dir = os.path.join(out_dir, 'details')
    os.makedirs(details_dir, exist_ok=True)

    dtl_img_path_lbl2det = _plot_details(image_root, cooked_diag, details_dir)
    dtl_img_path_negcnt_lst = _get_img_neg_cnts(dtl_img_path_lbl2det)

    # Sort and dump false positives and false negatives
    index_max = len(dtl_img_path_negcnt_lst)
    index_len_max = int(math.floor(math.log10(index_max)) + 1)

    fp_dir = os.path.join(out_dir, 'false_positives')
    fn_dir = os.path.join(out_dir, 'false_negatives')

    fp_prefix_fmt = '{{:0{}}}_fp_{{}}'.format(index_len_max)
    fn_prefix_fmt = '{{:0{}}}_fn_{{}}'.format(index_len_max)

    def fp_prefix_fn(i, negcnt): return fp_prefix_fmt.format(i, negcnt.fp_cnt)

    def fn_prefix_fn(i, negcnt): return fn_prefix_fmt.format(i, negcnt.fn_cnt)

    def fp_key_fn(dtl_img_path_negcnt): return -dtl_img_path_negcnt[1].fp_cnt

    def fn_key_fn(dtl_img_path_negcnt): return -dtl_img_path_negcnt[1].fn_cnt

    _dump_sorted_negcnt_lst(dtl_img_path_negcnt_lst, dst_dir=fp_dir, key_fn=fp_key_fn, prefix_fn=fp_prefix_fn)
    _dump_sorted_negcnt_lst(dtl_img_path_negcnt_lst, dst_dir=fn_dir, key_fn=fn_key_fn, prefix_fn=fn_prefix_fn)


def _dump_sorted_negcnt_lst(dtl_img_path_negcnt_lst, dst_dir, key_fn, prefix_fn):
    """
    Create sorted sym-links to the original detail image.

    dtl_img_path_negcnt_lst     [(detail image path,  NegCnts)]
    dst_dir                     destination directory
    key_fn                      callable((detail image path, NegCnts)) used to sort dtl_img_path_negcnt_lst
    prefix_fn                   callable(index, NegCnts) used to create the symlink prefix
    """
    os.makedirs(dst_dir, exist_ok=True)

    sorted_lst = sorted(dtl_img_path_negcnt_lst, key=key_fn)
    for i, (dtl_img_path, neg_cnt) in enumerate(sorted_lst, start=1):
        _, ext = os.path.splitext(dtl_img_path)
        ln_path = os.path.join(dst_dir, prefix_fn(i, neg_cnt) + ext)
        os.symlink(os.path.relpath(dtl_img_path, start=dst_dir), ln_path)


def _plot_details(image_root, cooked_diag, details_dir):
    """
    Return [(output detail image path,  {label: CookedDetection})]
    """
    ret = []
    for img_path in cooked_diag.iter_image_paths():
        lbl2det = cooked_diag.get_image_detection(img_path)
        img_real_path = os.path.join(image_root, img_path)
        detail_img = _plot_detection(img_real_path, lbl2det)

        out_path = os.path.join(details_dir, img_path)
        out_dir = os.path.dirname(out_path)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        cv2.imwrite(out_path, detail_img)
        ret.append((out_path, lbl2det))

    return ret


def _plot_detection(img_real_path, lbl2cdet):
    """
    Return an image.

    image_path  path to the image file
    lbl2cdet     {label: CookedDetection}
    """
    GND_TRUTH_COLOR = (0, 255, 0)
    FALSE_NEG_COLOR = (255, 0, 0)
    TRUE_POS_COLOR  = (199, 199, 0)
    FALSE_POS_COLOR = (0, 0, 255)

    GND_TRUTH_THICKNESS = 5
    OTHER_THICKNESS = 2

    canvas = cv2.imread(img_real_path, cv2.IMREAD_COLOR)

    def draw_xxyy_ary(ary, color, thickness):
        for x0, y0, x1, y1 in ary:
            x0, y0, x1, y1 = [int(z) for z in (x0, y0, x1, y1)]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thickness)

    def draw_xxyys_ary(ary, color, thickness):
        # Draw bboxes first, in case score string is overlaid by bbox edges
        for x0, y0, x1, y1, _ in ary:
            x0, y0, x1, y1 = [int(z) for z in (x0, y0, x1, y1)]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thickness)

        for x0, y0, _, _, score in ary:
            x0, y0 = [int(z) for z in (x0, y0)]
            s = '{:.2f}'.format(score)
            cv2.putText(canvas, s, (x0, y0 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 5)
            cv2.putText(canvas, s, (x0, y0 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    for cdet in lbl2cdet.values():
        draw_xxyy_ary(cdet.raw.annotations, GND_TRUTH_COLOR, GND_TRUTH_THICKNESS)
        draw_xxyy_ary(cdet.false_negatives, FALSE_NEG_COLOR, OTHER_THICKNESS)
        draw_xxyys_ary(cdet.true_positives, TRUE_POS_COLOR, OTHER_THICKNESS)
        draw_xxyys_ary(cdet.false_positives, FALSE_POS_COLOR, OTHER_THICKNESS)
        return canvas


# Negative counts of an image
# fp_cnt        false positives count
# fn_cnt        false negative count
NegCnts = namedtuple('NegCnts', ['fp_cnt', 'fn_cnt'])


def _get_img_neg_cnts(detail_img_path_lbl2cdet):
    """
    Return [(detail image path, NegCnts)]

    img_path_lbl2cdet   [(detail image path, {label: CookedDetection})]
    """
    ret = []

    for detail_img_path, lbl2cdet in detail_img_path_lbl2cdet:
        fp_cnt = 0
        fn_cnt = 0

        for cdet in lbl2cdet.values():
            assert isinstance(cdet, CookedDetection)
            fp_cnt += len(cdet.false_positives)
            fn_cnt += len(cdet.false_negatives)

        ret.append((detail_img_path, NegCnts(fp_cnt=fp_cnt, fn_cnt=fn_cnt)))

    return ret

import os
import numpy as np
import cv2

from ..eval import Diagnostic, ImageDetection


def plot_diag_detail(image_root, diag, out_dir):
    details_dir = os.path.join(out_dir, 'details')
    for img_path in diag.iter_image_paths():
        lbl2det = diag.get_image_detection(img_path)
        img_real_path = os.path.join(image_root, img_path)
        detail_img = _plot_detection(img_real_path, lbl2det)

        out_path = os.path.join(details_dir, img_path)
        out_dir = os.path.dirname(out_path)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        cv2.imwrite(out_path, detail_img)


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

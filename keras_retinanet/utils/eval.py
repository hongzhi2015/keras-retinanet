"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections_hl, draw_annotations

from collections import namedtuple, OrderedDict
import numpy as np
import os

import cv2
# import pickle


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, hl_score_threshold=0.36, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        hl_score_threshold: High-light detections whose scores are above this threshold.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

        # correct boxes for image scale
        detections[0, :, :4] /= scale

        # select scores from detections
        scores = detections[0, :, 4:]

        # select indices which have a score above the threshold
        indices = np.where(detections[0, :, 4:] > score_threshold)

        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = detections[0, indices[0][scores_sort], :4]
        image_scores     = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)
        image_predicted_labels = indices[1][scores_sort]

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), generator=generator, draw_label=False)
            draw_detections_hl(raw_image, detections[0, indices[0][scores_sort], :], generator=generator, draw_label=False)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_predicted_labels == label, :]

        print('{}/{}'.format(i, generator.size()), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i, generator.size()), end='\r')

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    hl_score_threshold=0.36,
    max_detections=100,
    save_path=None,
    diagnosis=False
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        hl_score_threshold: When the score confidence of a detection is above this threshold,
                            hight-light this detection in orange.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model,
                                         score_threshold=score_threshold,
                                         hl_score_threshold=hl_score_threshold,
                                         max_detections=max_detections,
                                         save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}
    recalls = {}
    precisions = {}
    return_scores = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision
        recalls[label] = recall
        precisions[label] = precision
        return_scores[label] = scores[indices]

    if diagnosis:
        return average_precisions, recalls, precisions, return_scores
    else:
        return average_precisions


class Diagnostic(object):
    # Detection from an image
    ImageDetection = namedtuple('_ImageDetection', [
        'annotations',
        'true_positives',
        'false_positives'])

    # Aggregated detections of a label
    LabelDetection = namedtuple('_LabelDetection', [
        'average_precision',
        'recalls',
        'precisions',
        'scores'])

    def __init__(self):
        # {image path: {label: ImageDetection}}
        self._img_dets = OrderedDict()

        # After freezing this becomes.
        # {label: LabelDetection}
        self._lbl_dets = None

    def add(self, img_path, label, annotations, true_positives, false_positives):
        """
        Add detection result of a label of an image.
        But not all statstics are updated, call flush() to update statistics.
        """
        if img_path not in self._img_dets:
            self._img_dets[img_path] = OrderedDict()

        assert label not in self._img_dets[img_path]
        self._img_dets[img_path][label] = self.ImageDetection(
            annotations=annotations,
            true_positives=true_positives,
            false_positives=false_positives)

    def _flush_by_label(self, label):
        """
        Calculate and aggregate statistics by label.
        Return LabelDetection.
        """
        anns_cnt = 0
        scores = []
        tp_indicators = []
        fp_indicators = []

        for _, label_det_m in self._img_dets.items():
            det = label_det_m[label]
            anns_cnt += det.annotations.shape[0]

            for each in det.true_positives:
                scores.append(each[4])
                tp_indicators.append(True)
                fp_indicators.append(False)

            for each in det.false_positives:
                scores.append(each[4])
                tp_indicators.append(False)
                fp_indicators.append(True)

        # Convert to numpy array
        scores = np.array(scores, dtype=np.float)
        tp_indicators = np.array(tp_indicators, dtype=np.int)
        fp_indicators = np.array(fp_indicators, dtype=np.int)

        # sort by score
        indices = np.argsort(-scores)
        sorted_scores = scores[indices]
        sorted_tp_indicators = tp_indicators[indices]
        sorted_fp_indicators = fp_indicators[indices]

        sorted_tp_cumsums = np.cumsum(sorted_tp_indicators)
        sorted_fp_cumsums = np.cumsum(sorted_fp_indicators)

        # compute recall and precision
        recalls = sorted_tp_cumsums / float(anns_cnt)
        precisions = sorted_tp_cumsums / np.maximum(sorted_tp_cumsums + sorted_fp_cumsums, np.finfo(np.float64).eps)

        # compute average precision
        ave_precision = _compute_ap(recalls, precisions)

        return self.LabelDetection(
            average_precision=ave_precision,
            recalls=recalls,
            precisions=precisions,
            scores=sorted_scores)

    def freeze(self):
        """
        Calculate and aggregate statistics.
        """
        assert self._lbl_dets is None, 'freeze is only allowed to call ONCE'
        self._lbl_dets = OrderedDict()

        # Collect labels
        labels = set()
        for _, label_det_m in self._img_dets.items():
            labels.update(label_det_m.keys())

        labels = sorted(labels)

        # Aggregate statistics
        for label in labels:
            lbl_det = self._flush_by_label(label)
            self._lbl_dets[label] = lbl_det
            if True:
                print()
                print('############################')
                print('NEW STATISTICS')
                print('############################')
                print('### label:', label)
                print('### average precision:', lbl_det.average_precision)
                print('### recalls:', lbl_det.recalls)
                print('### precisions:', lbl_det.precisions)
                print('### scores:', lbl_det.scores)
                print()
        pass


def _collect_diags(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    hl_score_threshold=0.36,
    max_detections=100,
    save_path=None
):
    """
    Collect diagnostic data from an evaluation set from given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        hl_score_threshold: When the score confidence of a detection is above this threshold,
                            hight-light this detection in orange.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.

    # Returns
        Diagnostic
    """
    ret = Diagnostic()

    # gather all detections and annotations
    all_detections     = _get_detections(generator, model,
                                         score_threshold=score_threshold,
                                         hl_score_threshold=hl_score_threshold,
                                         max_detections=max_detections,
                                         save_path=save_path)
    all_annotations    = _get_annotations(generator)

    for i in range(generator.size()):
        for label in range(generator.num_classes()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]

            detected_annotations = []
            true_positives  = []
            false_positives = []

            for d in detections:
                if annotations.shape[0] == 0:
                    false_positives.append(d)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    true_positives.append(d)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(d)

            ret.add(
                img_path=generator.image_path(i),
                label=label,
                annotations=annotations,
                true_positives=true_positives,
                false_positives=false_positives)

    ret.freeze()
    return ret


def evaluate_diag(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    hl_score_threshold=0.36,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model with various diagnostic data dumped.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        hl_score_threshold: When the score confidence of a detection is above this threshold,
                            hight-light this detection in orange.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores, when diagnosis is False.
        [Diagnostic], when diagnosis is True.
    """
    xx = _collect_diags(
        generator=generator,
        model=model,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        hl_score_threshold=hl_score_threshold,
        max_detections=max_detections,
        save_path=save_path)

    # gather all detections and annotations
    all_detections     = _get_detections(generator, model,
                                         score_threshold=score_threshold,
                                         hl_score_threshold=hl_score_threshold,
                                         max_detections=max_detections,
                                         save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}
    recalls = {}
    precisions = {}
    return_scores = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision
        recalls[label] = recall
        precisions[label] = precision
        return_scores[label] = scores[indices]

    if True:
        print()
        print('@@@@@@@@@@@@@@@@@@@@@')
        print('Old statistics')
        print('@@@@@@@@@@@@@@@@@@@@@')
        print('@@ average_precisions', average_precisions)
        # print('@@ image_names', generator.image_names)
        print('@@ recalls', recalls)
        print('@@ precisions', precisions)
        print('@@ scores', return_scores)
        print()
    return average_precisions, generator.image_names, recalls, precisions, return_scores

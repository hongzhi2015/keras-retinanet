from ..anchors import compute_overlap
from ..visualization import draw_detections, draw_annotations

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


# def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
#     """ Get the detections from the model using the generator.
#
#     The result is a list of lists such that the size is:
#         all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
#
#     # Arguments
#         generator       : The generator used to run images through the model.
#         model           : The model to run on the images.
#         score_threshold : The score confidence threshold to use.
#         max_detections  : The maximum number of detections to use per image.
#         save_path       : The path to save the images with visualized detections to.
#     # Returns
#         A list of lists containing the detections for each image in the generator.
#     """
#     all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
#
#     for i in range(generator.size()):
#         raw_image    = generator.load_image(i)
#         image        = generator.preprocess_image(raw_image.copy())
#         image, scale = generator.resize_image(image)
#
#         # run network
#         _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
#
#         # clip to image shape
#         detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
#         detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
#         detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
#         detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])
#
#         # correct boxes for image scale
#         detections[0, :, :4] /= scale
#
#         # select scores from detections
#         scores = detections[0, :, 4:]
#
#         # select indices which have a score above the threshold
#         indices = np.where(detections[0, :, 4:] > score_threshold)
#
#         # select those scores
#         scores = scores[indices]
#
#         # find the order with which to sort the scores
#         scores_sort = np.argsort(-scores)[:max_detections]
#
#         # select detections
#         image_boxes      = detections[0, indices[0][scores_sort], :4]
#         image_scores     = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
#         image_detections = np.append(image_boxes, image_scores, axis=1)
#         image_predicted_labels = indices[1][scores_sort]
#
#         if save_path is not None:
#             draw_annotations(raw_image, generator.load_annotations(i), generator=generator, draw_label=False)
#             draw_detections(raw_image, detections[0, indices[0][scores_sort], :], generator=generator, draw_label=False)
#             cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
#
#         # copy detections to all_detections
#         for label in range(generator.num_classes()):
#             all_detections[i][label] = image_detections[image_predicted_labels == label, :]
#
#         print('{}/{}'.format(i, generator.size()), end='\r')
#
#     return all_detections


###################################################
# Tried to trace how the original algorithm works #
###################################################
#
# In [112]: detections
# Out[112]:
# array([[[ 1.  ,  2.  ,  3.  ,  4.  ,  0.3 ,  0.9 ],
#         [ 3.  ,  4.  ,  5.  ,  6.  ,  0.4 ,  0.7 ],
#         [ 7.  ,  8.  ,  9.  ,  0.  ,  0.5 ,  0.55]]])
#
# In [113]: scores = detections[0, :, 4:]
#
# In [114]: scores
# Out[114]:
# array([[ 0.3 ,  0.9 ],
#        [ 0.4 ,  0.7 ],
#        [ 0.5 ,  0.55]])
#
# In [128]: indices = np.where(scores > 0.35)
#
# In [129]: indices
# Out[129]: (array([0, 1, 1, 2, 2]), array([1, 0, 1, 0, 1]))
# COMMENT: The 1st array is for both detections and scores.
#          The 2nd array is for the offset to the label0 score of each detection.
#
#
# In [130]: scores = scores[indices]
#
# In [132]: scores
# Out[132]: array([ 0.9 ,  0.4 ,  0.7 ,  0.5 ,  0.55])
# COMMENT: Now each score in the scores corresponds to indices[0] or indices[1]
#
#
# In [133]: scores_sort = np.argsort(-scores)
# COMMENT: indices[0] and indices[1] are indirectly sorted by scores sort.
#
# In [134]: scores_sort
# Out[134]: array([0, 2, 4, 3, 1])
#
#
# In [135]: image_boxes      = detections[0, indices[0][scores_sort], :4]
#
# In [136]: image_boxes
# Out[136]:
# array([[ 1.,  2.,  3.,  4.],
#        [ 3.,  4.,  5.,  6.],
#        [ 7.,  8.,  9.,  0.],
#        [ 7.,  8.,  9.,  0.],
#        [ 3.,  4.,  5.,  6.]])
#
# In [139]: indices[0][scores_sort]
# Out[139]: array([0, 1, 2, 2, 1])
#
# In [140]: image_scores     = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
#
# In [141]: image_scores
# Out[141]:
# array([[ 0.9 ],
#        [ 0.7 ],
#        [ 0.55],
#        [ 0.5 ],
#        [ 0.4 ]])
#
# In [142]: image_detections = np.append(image_boxes, image_scores, axis=1)
#
# In [144]: image_detections
# Out[144]:
# array([[ 1.  ,  2.  ,  3.  ,  4.  ,  0.9 ],
#        [ 3.  ,  4.  ,  5.  ,  6.  ,  0.7 ],
#        [ 7.  ,  8.  ,  9.  ,  0.  ,  0.55],
#        [ 7.  ,  8.  ,  9.  ,  0.  ,  0.5 ],
#        [ 3.  ,  4.  ,  5.  ,  6.  ,  0.4 ]])
#
#
# In [145]: image_predicted_labels = indices[1][scores_sort]
# In [146]: image_predicted_labels
# Out[146]: array([1, 1, 1, 0, 0])


def _get_detections(generator, model, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
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

        if False:
            print('### model detections:', detections.shape, detections.dtype)

        # select scores from detections
        scores = detections[0, :, 4:]

        if False:
            print('### model scores:', scores.shape, scores.dtype)

        # FIXME: REMOVE score_threshold
        score_threshold = 0.0
        # select indices which have a score above the threshold
        indices = np.where(detections[0, :, 4:] > score_threshold)

        if False:
            print('### detections[0, :, 4:]', detections[0, :, 4:].shape)
            print('### score indices:', indices)

        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        if False:
            print('### scores', scores.shape, scores.dtype)
            print('### scores_sort', scores_sort.shape, scores_sort.dtype)
            print('### indices[0]', indices[0].shape, indices[0].dtype)

        # select detections
        image_boxes      = detections[0, indices[0][scores_sort], :4]
        image_scores     = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)
        image_predicted_labels = indices[1][scores_sort]

        if False:
            print('### indices[1][scores_sort]', indices[1][scores_sort].shape, indices[1][scores_sort].dtype)
            print('### image_bboxes', image_boxes.shape, image_boxes.dtype)
            print('### image_scores', image_scores.shape, image_scores.dtype)
            print('### image_detections', image_detections.shape, image_detections.dtype)
            print('### image_predicted_labels', image_predicted_labels.shape, image_predicted_labels.dtype)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), generator=generator, draw_label=False)
            draw_detections(raw_image, detections[0, indices[0][scores_sort], :], generator=generator, draw_label=False)
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_predicted_labels == label, :]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


class ModelDetections:
    """
    Hold detections from a model directly.
    """

    class EvalSample(namedtuple('_EvalSample', ['path', 'annotations', 'detections'])):
        """
        path            path of an image
        annotations     ndarray(shape=(N, 5))
                        N annotations
                        each annotation is in [x0, y0, x1, y1, LABEL]
        detections      ndarray(shape=(1, M, 4 + label number))
                        M annotations
                        each detection is in [x0, y0, x1, y1, score0, ... score_LAST]
        """
        __slots__ = ()

        def __new__(cls, path, annotations, detections):
            assert len(annotations.shape) == 2 and annotations.shape[-1] == 5
            assert len(detections.shape) == 3
            assert detections.shape[-1] > 4
            return super().__new__(cls, path, annotations, detections)

        def num_classes(self):
            return self.detections.shape[-1] - 4

    def __init__(self, num_classes):
        """
        num_classes     number of classes
        """
        self._num_classes = num_classes
        # {under root image path: EvalSample}
        self._ess = []
        pass

    def add(self, path, anns, dets):
        """
        Add raw detections for an image.

        path    str         image path under root
        anns    ndarray     annotations
        dets    ndarray     raw detections from the model
        """
        es = self.EvalSample(path, anns, dets)
        assert es.num_classes() == self._num_classes
        self._ess.append(self.EvalSample(path, anns, dets))

    def num_classes(self):
        return self._num_classes

    def get_all_detections(self, score_thresh):
        """
        Return OrderedDict {ur img path: [DETETIONS_FOR_A_LABEL]}
        DETECTIONS_FOR_A_LABEL := numpy.ndarray(shape=(N detections, 5))
        A label is the index of [DETETIONS_FOR_A_LABEL]

        score_thresh    The score confidence threshold to use.
        """
        all_detections = OrderedDict()

        for es in self._ess:
            detections = es.detections

            # select scores from detections
            scores = detections[0, :, 4:]

            # select indices which have a score above the threshold
            indices = np.where(detections[0, :, 4:] > score_thresh)

            # select those scores
            scores = scores[indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)

            # select detections
            image_boxes      = detections[0, indices[0][scores_sort], :4]
            image_scores     = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
            image_detections = np.append(image_boxes, image_scores, axis=1)
            image_predicted_labels = indices[1][scores_sort]

            # copy detections to all_detections
            all_detections[es.path] = [image_detections[image_predicted_labels == label, :]
                                       for label in range(self._num_classes)]

        return all_detections

    def get_all_annotations(self):
        """
        Return OrderedDict {ur img path: [ANNOTATIONS_FOR_A_LABEL]}
        ANNOTATIONS_FOR_A_LABEL := numpy.ndarray(shape=(N detections, 4))
        A label is the index of [ANNOTATIONS_FOR_A_LABEL]
        """
        all_annotations = OrderedDict()

        for es in self._ess:
            annotations = es.annotations
            all_annotations[es.path] = [annotations[annotations[:, 4] == label, :4].copy()
                                        for label in range(self._num_classes)]

        return all_annotations


def _get_model_detections(generator, model):
    """
    Return ModelDetections instance.
    """
    ret = ModelDetections(generator.num_classes())
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

        annotations = generator.load_annotations(i)
        ur_img_path = os.path.relpath(generator.image_path(i), start=generator.base_dir)
        ret.add(ur_img_path, annotations, detections)

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return ret


def _get_all_detections(generator, model):
    """ Get the all detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
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

        # select detections
        bboxes = detections[0, :, :4]
        scores = detections[0, :, 4:]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = np.append(bboxes, scores[np.newaxis, :, label])

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

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


###########################
# Detection from an image #
###########################

class RawDetection(namedtuple('_RawDetection', ['annotations', 'bboxes'])):
    """
    Raw detections from the model.
    """
    __slots__ = ()

    def __new__(cls, annotations, bboxes):
        """
        annotations   ndarray(shape=(N, 4))
        bboxes        ndarray(shape=(K, 5)), 4 coordinates and one score that belongs to some category
        """
        assert len(annotations.shape) == 2 and annotations.shape[1] == 4
        assert len(bboxes.shape) == 2 and bboxes.shape[1] == 5
        return super().__new__(cls, annotations, bboxes)

    def in_score_range(self, lb, ub):
        """
        Return RawDetection with scores of bboxes in [lb, ub].
        Both boundaries are inclusive.
        """
        scores = self.bboxes[:, 4]
        new_bboxes = self.bboxes[(lb <= scores) & (scores <= ub)]
        return RawDetection(annotations=self.annotations, bboxes=new_bboxes)


class CookedDetection(namedtuple('_CookedDetection', [
     # Raw Detection
     'raw',
     # [ndarray(x0, y0, x1, y1, score)]
     'true_positives',
     # [ndarray(x0, y0, x1, y1, score)]
     'false_positives',
     # [ndarray(x0, y0, x1, y1)]
     'false_negatives'])):

    __slots__ = ()

    def __new__(cls, raw, iou_thresh, score_range):
        """
        raw             RawDetection
        iou_thresh      float   IoU threshold
        score_range     (inclusive lower score in float, inclusive upper score in float)
        """
        assert isinstance(raw, RawDetection)
        raw = raw.in_score_range(*score_range)
        tps, fps, fns = cls._split_bboxes(raw, iou_thresh)
        return super().__new__(cls, raw=raw,
                               true_positives=tps, false_positives=fps, false_negatives=fns)

    @staticmethod
    def _split_bboxes(raw, iou_thresh):
        """
        Return ([true positive bbox], [false positive bbox], [false negative bbox])
        """
        tps = []
        fps = []
        fns = []
        detected_ann_idcs = []

        if raw.annotations.shape[0] == 0:
            return tps, fps, fns

        # Get true positives and false positives
        for b in raw.bboxes:
            overlaps = compute_overlap(np.expand_dims(b, axis=0), raw.annotations)
            assigned_ann_idx = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_ann_idx]

            assert iou_thresh > 0, 'the condition check requires iou_thresh be postitive'
            if max_overlap >= iou_thresh and assigned_ann_idx not in detected_ann_idcs:
                tps.append(b)
                detected_ann_idcs.append(assigned_ann_idx)
            else:
                fps.append(b)

        # Get false netatives
        fns.extend(raw.annotations[i]
                   for i in range(raw.annotations.shape[0])
                   if i not in detected_ann_idcs)

        return tps, fps, fns


# Aggregated detections of a label
# Note: Put it just under the module for pickling purpose
LabelDetection = namedtuple('LabelDetection', [
    'average_precision',
    'recalls',
    'precisions',
    'scores'])


class RawDiagnostic(object):
    def __init__(self):
        # {image path: {label: RawDetection}}
        self._dets = OrderedDict()

    def add(self, ur_img_path, label, annotations, bboxes):
        """
        Add detection result of a label of an image.

        ur_img_path     The image path under the image root dir.
        """
        if ur_img_path not in self._dets:
            self._dets[ur_img_path] = OrderedDict()

        assert label not in self._dets[ur_img_path]
        self._dets[ur_img_path][label] = RawDetection(annotations=annotations, bboxes=bboxes)

    def iter_image_paths(self):
        return self._dets.keys()

    def get_label2detections(self, img_path):
        """
        Return {label: RawDetection} with given img_path.
        """
        return self._dets[img_path]


def get_raw_diag(model_dets, score_thresh):
    """
    Return RawDiagnostic from ModelDetections instace.

    model_dets  ModelDetections
    """
    ret = RawDiagnostic()

    # gather all detections and annotations
    all_detections  = model_dets.get_all_detections(score_thresh)
    all_annotations = model_dets.get_all_annotations()
    assert all_detections.keys() == all_annotations.keys()

    for img_path in all_detections.keys():
        for label in range(model_dets.num_classes()):
            bboxes = all_detections[img_path][label]
            anns = all_annotations[img_path][label]
            ret.add(ur_img_path=img_path,
                    label=label,
                    annotations=anns,
                    bboxes=bboxes)
    return ret



class CookedDiagnostic(object):
    def __init__(self, raw_diag, iou_thresh, score_range):
        """
        raw_diag        RawDiagnostic instance
        iou_thresh      float   IoU threshold
        score_range     (inclusive lower score in float, inclusive upper score in float)
        """
        assert isinstance(raw_diag, RawDiagnostic)

        # {image path: {label: CookedDetection}}
        self._dets = self._get_cooked_detections(raw_diag, iou_thresh, score_range)

        # {label: LabelDetection}
        self._lbl_dets = self._get_label_detections(self._dets)
        pass

    @staticmethod
    def _get_cooked_detections(raw_diag, iou_thresh, score_range):
        """
        Return {image path: {label: CookedDetection}}
        """
        assert isinstance(raw_diag, RawDiagnostic)
        ret = OrderedDict()
        for img_path in raw_diag.iter_image_paths():
            raw_lbl2dets = raw_diag.get_label2detections(img_path)
            for lbl, raw_det in raw_lbl2dets.items():
                cooked_det = CookedDetection(raw=raw_det, iou_thresh=iou_thresh, score_range=score_range)
                assert img_path not in ret
                if img_path not in ret:
                    ret[img_path] = OrderedDict()

                assert lbl not in ret[img_path]
                ret[img_path][lbl] = cooked_det

        return ret

    @classmethod
    def _get_label_detections(cls, dets):
        """
        Return {label: LabelDetection}

        dets    {image path: {label: CookedDetection}}
        """
        ret = OrderedDict()

        # Collect labels
        labels = set()
        for lbl2cdet in dets.values():
            labels.update(lbl2cdet.keys())

        labels = sorted(labels)

        # Aggregate statistics
        for label in labels:
            lbl_det = cls._aggregate_by_label(dets, label)
            ret[label] = lbl_det

        return ret

    @staticmethod
    def _aggregate_by_label(dets, label):
        """
        Aggregate statistics by label.
        Return LabelDetection.

        dets    {image path: {label: CookedDetection}}
        label   a label from dets
        """
        anns_cnt = 0
        scores = []
        tp_indicators = []
        fp_indicators = []

        for lbl2cdet in dets.values():
            cdet = lbl2cdet[label]
            anns_cnt += cdet.raw.annotations.shape[0]

            for tp in cdet.true_positives:
                scores.append(tp[4])
                tp_indicators.append(True)
                fp_indicators.append(False)

            for fp in cdet.false_positives:
                scores.append(fp[4])
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

        return LabelDetection(
            average_precision=ave_precision,
            recalls=recalls,
            precisions=precisions,
            scores=sorted_scores)

    #########################
    # High-level Statistics #
    #########################
    def get_labels(self):
        """
        Return labels in [label]
        """
        return list(self._lbl_dets.keys())

    def get_label_detection(self, label):
        """
        Return LabelDetection instance with given label.
        """
        return self._lbl_dets[label]

    def get_mAP(self):
        """
        Return Mean Average Precision of all detections.
        """
        aps = [d.average_precision for d in self._lbl_dets.values()]
        return sum(aps) / len(aps)

    #####################
    # Detection Details #
    #####################
    def iter_image_paths(self):
        return self._dets.keys()

    def get_image_detection(self, img_path):
        """
        Return {label: CookedDetection} with given img_path.
        """
        return self._dets[img_path]


class Diagnostic(object):
    def __init__(self):
        # {image path: {label: ImageDetection}}
        self._img_dets = OrderedDict()

        # After freezing this becomes.
        # {label: LabelDetection}
        self._lbl_dets = None

    def _assert_not_freeze(self):
        """
        Assert when freeze has been called.
        """
        assert self._lbl_dets is None, 'Do not do this operation after calling freeze()'

    def _assert_freeze(self):
        """
        Assert when freeze has not been called.
        """
        assert self._lbl_dets is not None, 'Call freeze() to collect label aggregated data'

    def add(self, ur_img_path, label, annotations, true_positives, false_positives):
        """
        Add detection result of a label of an image.
        But not all statstics are updated, call flush() to update statistics.

        ur_img_path     The image path under the image root dir.
        """
        self._assert_not_freeze()

        if ur_img_path not in self._img_dets:
            self._img_dets[ur_img_path] = OrderedDict()

        assert label not in self._img_dets[ur_img_path]
        self._img_dets[ur_img_path][label] = ImageDetection(
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

        return LabelDetection(
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
        pass

    #########################
    # High-level Statistics #
    #########################
    def get_labels(self):
        """
        Return labels in [label]
        """
        self._assert_freeze()
        return list(self._lbl_dets.keys())

    def get_label_detection(self, label):
        """
        Return LabelDetection instance with given label.
        """
        self._assert_freeze()
        return self._lbl_dets[label]

    def get_mAP(self):
        """
        Return Mean Average Precision of all detections.
        """
        self._assert_freeze()
        aps = [d.average_precision for d in self._lbl_dets.values()]
        return sum(aps) / len(aps)

    #####################
    # Detection Details #
    #####################
    def iter_image_paths(self):
        return self._img_dets.keys()

    def get_image_detection(self, img_path):
        """
        Return {label: ImageDetection} with given img_path.
        """
        return self._img_dets[img_path]


# def evaluatex(
#     generator,
#     model,
#     iou_threshold=0.5,
#     score_threshold=0.05,
#     max_detections=100,
#     save_path=None
# ):
#     """ Evaluate a given dataset using a given model with various diagnostic data dumped.
#
#     # Arguments
#         generator       : The generator that represents the dataset to evaluate.
#         model           : The model to evaluate.
#         iou_threshold   : The threshold used to consider when a detection is positive or negative.
#         score_threshold : The score confidence threshold to use for detections.
#         max_detections  : The maximum number of detections to use per image.
#         save_path       : The path to save images with visualized detections to.
#     # Returns
#         Diagnostic
#     """
#     ret = Diagnostic()
#
#     # gather all detections and annotations
#     all_detections     = _get_detections(generator, model,
#                                          score_threshold=score_threshold,
#                                          max_detections=max_detections,
#                                          save_path=save_path)
#     all_annotations    = _get_annotations(generator)
#
#     for i in range(generator.size()):
#         for label in range(generator.num_classes()):
#             detections           = all_detections[i][label]
#             annotations          = all_annotations[i][label]
#
#             detected_annotations = []
#             true_positives  = []
#             false_positives = []
#
#             for d in detections:
#                 if annotations.shape[0] == 0:
#                     false_positives.append(d)
#                     continue
#
#                 overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
#                 assigned_annotation = np.argmax(overlaps, axis=1)
#                 max_overlap         = overlaps[0, assigned_annotation]
#
#                 if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
#                     true_positives.append(d)
#                     detected_annotations.append(assigned_annotation)
#                 else:
#                     false_positives.append(d)
#
#             ret.add(
#                 ur_img_path=os.path.relpath(generator.image_path(i), start=generator.base_dir),
#                 label=label,
#                 annotations=annotations,
#                 true_positives=true_positives,
#                 false_positives=false_positives)
#
#     ret.freeze()
#     return ret


def evaluate(
    generator,
    model,
    save_path=None
):
    """ Evaluate a given dataset using a given model with various diagnostic data dumped.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        save_path       : The path to save images with visualized detections to.
    # Returns
        Diagnostic
    """
    return _get_model_detections(generator, model)

    raw_diag = RawDiagnostic()

    # gather all detections and annotations
    all_detections  = _get_detections(generator, model, save_path=save_path)
    all_annotations = _get_annotations(generator)

    if True:
        print()
        print('#### all detections', type(all_detections))
        print('#### all_detections[0]', type(all_detections[0]))
        print('#### all_detections[0][0]', type(all_detections[0][0]), all_detections[0][0].shape)
        print()

    for i in range(generator.size()):
        ur_img_path = os.path.relpath(generator.image_path(i), start=generator.base_dir)

        for label in range(generator.num_classes()):
            bboxes = all_detections[i][label]
            anns   = all_annotations[i][label]
            raw_diag.add(ur_img_path=ur_img_path,
                         label=label,
                         annotations=anns,
                         bboxes=bboxes)

    return raw_diag

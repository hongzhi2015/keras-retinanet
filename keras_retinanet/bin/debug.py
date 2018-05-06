#!/usr/bin/env python

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

import argparse
import os
import sys
import cv2
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.transform import random_transform_generator
from ..utils.visualization import draw_annotations, draw_boxes
from ..utils.anchors import compute_overlap, greedy_nms

def create_generator(args):
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        generator = CocoGenerator(
            args.coco_path,
            args.coco_set,
            transform_generator=transform_generator
        )
    elif args.dataset_type == 'pascal':
        generator = PascalVocGenerator(
            args.pascal_path,
            args.pascal_set,
            transform_generator=transform_generator
        )
    elif args.dataset_type == 'csv':
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            base_dir=args.image_dir,
            image_min_side=960,
            image_max_side=1280,
            positive_overlap=args.positive_overlap,
            negative_overlap=args.negative_overlap,
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path',  help='Path to dataset directory (ie. /tmp/COCO).')
    coco_parser.add_argument('--coco-set', help='Name of the set to show (defaults to val2017).', default='val2017')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--pascal-set',  help='Name of the set to show (defaults to test).', default='test')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('-l', '--loop', help='Loop forever, even if the dataset is exhausted.', action='store_true')
    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--negative-anchors', help='Show negative anchors on the image.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')

    parser.add_argument('--image-dir', help='where images are.', required=True)
    parser.add_argument('--positive-overlap', help='positive iou', default = 0.5, type=float)
    parser.add_argument('--negative-overlap', help='negative iou', default = 0.4, type=float)

    return parser.parse_args(args)

def pretty_print_box(title, boxes):
  b = boxes.astype(np.int)
  b[:,2] = b[:, 2] - b[:, 0]
  b[:,3] = b[:, 3] - b[:, 1]
  print(title, b)

def run(generator, args):
    # display images, one at a time
    for i in range(generator.size()):
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)

        # apply random transformations
        if args.random_transform:
            image, annotations = generator.random_transform_group_entry(image, annotations)

        # resize the image and annotations
        if args.resize:
            image, image_scale = generator.resize_image(image)
            annotations[:, :4] *= image_scale

        labels, boxes, anchors = generator.anchor_targets(image.shape, annotations, generator.num_classes())
        # draw anchors on the image
        if args.anchors:
            draw_boxes(image, anchors[np.max(labels, axis=1) == 1], (255, 255, 0), thickness=1)
            pretty_print_box('positives', anchors[np.max(labels, axis=1) == 1])

        if args.negative_anchors:
            # draw negative anchors that has overlap with a annotation
            overlaps             = compute_overlap(anchors, annotations[:, :4])
            negative = np.max(labels, axis=1) == 0
            has_overlap = np.max(overlaps, axis=1) > 0.001
            selected = np.logical_and(has_overlap, negative)
            negative_anchors = anchors[selected]
            nms_selected = greedy_nms(negative_anchors, np.max(overlaps[selected, :], axis=1), iou_threshold=0.3)
            negative_nms = negative_anchors[nms_selected]
            # debug
            # show small boxes only
            negative_nms = np.array(sorted(negative_nms.tolist(), key = lambda x: (x[3]-x[1])*(x[2]-x[0]))[:100])
            # debug
            draw_boxes(image, negative_nms, (76, 0, 200), thickness=1)
            pretty_print_box('negatives', negative_nms)

        # draw annotations on the image
        if args.annotations:
            # draw annotations in red
            draw_annotations(image, annotations, color=(0, 0, 255), generator=generator)

            # draw regressed anchors in green to override most red annotations
            # result is that annotations without anchors are red, with anchors are green
            draw_boxes(image, boxes[np.max(labels, axis=1) == 1], (0, 255, 0))
            pretty_print_box('annotations', annotations)

        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            return False
    return True


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generator
    generator = create_generator(args)

    # create the display window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    if args.loop:
        while run(generator, args):
            pass
    else:
        run(generator, args)

if __name__ == '__main__':
    main()

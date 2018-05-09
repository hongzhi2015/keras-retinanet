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
import pickle

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.keras_version import check_keras_version
from ..utils.evalx import get_eval_detections
from ..models.resnet import custom_objects


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    if args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            base_dir=args.image_dir,
            # No limit on input image size
            image_min_side=None,
            image_max_side=None,
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    parser     = argparse.ArgumentParser(
        description='Evaluation script for a RetinaNet network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',             help='Path to RetinaNet model.')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--image_dir',       help='where images are.', required=True)
    parser.add_argument('--output_metrics',  help='save the precision recalls out', required=True)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    # start evaluation
    eval_dets = get_eval_detections(generator, model)

    # print evaluation
    # ave_precs = []
    # for label in raw_diag.get_labels():
    #     lbl_det = raw_diag.get_label_detection(label)
    #     this_ave_prec = lbl_det.average_precision
    #     ave_precs.append(this_ave_prec)
    #     print(generator.label_to_name(label), '{:.4f}'.format(this_ave_prec))

    # print('mAP: {:.4f}'.format(sum(ave_precs) / len(ave_precs)))

    if args.output_metrics is not None:
        # In case, the dir of output metrics dir does not exist.
        os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
        with open(args.output_metrics, 'wb') as handle:
            pickle.dump(eval_dets, handle, protocol=4)


if __name__ == '__main__':
    main()

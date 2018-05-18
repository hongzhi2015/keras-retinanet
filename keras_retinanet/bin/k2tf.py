#!/usr/bin/env python

# First of all, disable tensorflow from using GPU.
# Because only model convertion is done.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys # noqa: E402
import argparse # noqa: E402

import keras # noqa: E402
from keras import backend as K # noqa: E402
import tensorflow as tf # noqa: E402
from tensorflow.python.framework import tensor_shape, graph_util # noqa: E402
from tensorflow.python.platform import gfile # noqa: E402

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin # noqa: W0611
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..utils.keras_version import check_keras_version # noqa: E402
from ..models.resnet import custom_objects # noqa: E402


def get_session():
    config = tf.ConfigProto()
    return tf.Session(config=config)


def k2tf(k_model_path, tf_model_path):
    """
    Convert Keras model to TensorFlow path.

    k_model_path    Keras model path
    t_model_path    TensorFlow model path
    """
    keras.backend.tensorflow_backend.set_session(get_session())
    model = keras.models.load_model(k_model_path, custom_objects=custom_objects)

    K.set_learning_phase(0)  # to get rid of learning rate and drop out
    sess = K.get_session()
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),
                                                                 [model.outputs[-1].name.replace(':0', '')])
    with gfile.FastGFile(tf_model_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Convert Keras model to TensorFlow model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-k', '--keras', type=str, required=True, help='Path to input Keras model')
    parser.add_argument('-t', '--tensorflow', type=str, required=True, help='Path to output TensorFlow model')
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args if args is not None else sys.argv[1:])
    # make sure keras is the minimum required version
    check_keras_version()
    k2tf(args.keras, args.tensorflow)


if __name__ == '__main__':
    main()

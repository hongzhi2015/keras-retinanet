import tensorflow as tf
import cv2
import sys
import numpy as np

def create_tf_graph(sess, model_path):
  with open(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def resize_image(img, min_side=960, max_side=1280):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, wich can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

if __name__ == '__main__':
  model_file = sys.argv[1]
  threshold = float(sys.argv[2])
  image_file = sys.argv[3]

  sess = tf.Session()
  create_tf_graph(sess, model_file)

  image = cv2.imread(image_file).astype(np.float32)
  image[..., 0] -= 103.939
  image[..., 1] -= 116.779
  image[..., 2] -= 123.68

  input_tensor = sess.graph.get_tensor_by_name('input_1:0')
  output_tensor = sess.graph.get_tensor_by_name('nms/ExpandDims:0')

  image, scale = resize_image(image)
  detections = sess.run(output_tensor, {input_tensor: np.expand_dims(image, axis=0)})

  # compute predicted labels and scores
  predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
  scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

  # correct for image scale
  detections[0, :, :4] /= scale

  for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
    if score < threshold:
        continue
    b = detections[0, idx, :4].astype(int)
    print(score, b)

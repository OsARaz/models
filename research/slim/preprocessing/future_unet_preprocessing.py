from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def preprocess_for_train(image, mask, height, width,
                         scope=None,
                         add_image_summaries=True):
    """preprocess one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different
    transformations applied to the image.

    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      scope: Optional scope for name_scope.
      add_image_summaries: Enable image summaries.
    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
      TODO: *OPTIONAL* this is a basic preprocessing function that doesn't augment the input feel free to augment it,
      TODO: *OPTIONAL* use functions from tf.image, or use other preprocessing modules found in this directory as reference
      Please start from this basic function and train with it before trying to augment this will make your lives much easier
    """
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected.
    if height and width:
        image = tf.expand_dims(image, 0)
        mask = tf.expand_dims(mask, 0)
        resized_image = tf.image.resize_images(image, [height, width])

        resized_mask = tf.image.resize_images(mask, [height, width],
                                              1)  # 1 stands for resize nearest neighbour TODO: think, why use nearest?
        resized_image = tf.squeeze(resized_image)
        resized_mask = tf.squeeze(resized_mask)


    distorted_image = tf.subtract(resized_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    distorted_mask = tf.cast(resized_mask / 255, tf.uint8)
    return distorted_image, distorted_mask


def preprocess_for_eval(image, mask, height, width, add_image_summaries=False, scope=None):
    # in our case eval preprocess is the same as train
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected.
    if height and width:
        image = tf.expand_dims(image, 0)
        mask = tf.expand_dims(mask, 0)
        resized_image = tf.image.resize_images(image, [height, width])

        resized_mask = tf.image.resize_images(mask, [height, width],
                                              1)  # 1 stands for resize nearest neighbour TODO: think, why use nearest?
        resized_image = tf.squeeze(resized_image)
        resized_mask = tf.squeeze(resized_mask)

    distorted_image = tf.subtract(resized_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    distorted_mask = tf.cast(resized_mask/255, tf.uint8)

    return distorted_image, distorted_mask


def preprocess_image(image, height, width,mask,
                     is_training=False,
                     add_image_summaries=True, scope=None):
    """Pre-process one image for training or evaluation.

    Args:
      image: 3-D Tensor [height, width, channels] with the image. If dtype is
        tf.float32 then the range should be [0, 1], otherwise it would converted
        to tf.float32 assuming that the range is [0, MAX], where MAX is largest
        positive representable number for int(8/16/32) data type (see
        `tf.image.convert_image_dtype` for details).
      height: integer, image expected height.
      width: integer, image expected width.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations.
      add_image_summaries: Enable image summaries.

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """
    if is_training:
        return preprocess_for_train(image, mask, height, width, scope, add_image_summaries)
    else:
        return preprocess_for_eval(image, mask, height, width, add_image_summaries, scope)

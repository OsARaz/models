from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_utils


from tensorflow.contrib import slim



@slim.add_arg_scope
def encoder_block(inputs,
               depth,
               stride,
               scope=None):
  """encoder block for unet ususaly repeated 4 times

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the encoder unit output
    stride: The encoder unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the encoder unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
      TODO: write the encoder block for unet
  Returns:
    The encoder unit's output.
  """
  output=input
  return output



@slim.add_arg_scope
def decoder_block(inputs,
               depth,
               stride,
               scope=None):
  """decoder block for unet ususaly repeated 4 times

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the decoder unit output
    stride: The decoder unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the decoder unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
      TODO: write the decoder block for unet
  Returns:
    The decoder unit's output.
  """
  output=input
  return output


def future_unet_arg_scope(**kwargs):
  """Default arg scope for the PNASNet Large ImageNet model."""
  with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def future_unet(inputs,num_classes,reuse=None,scope=None,**kwargs):
    # TODO: finish this function so that it returns your network (the last tensor in your network, in net)
    # in addition it is recommended to add relevant tensors to the end_points_collection,
    # this for example may help with extracting the tensors in the encoder used by the decoder (skip connections)
    net=slim.conv2d(inputs,num_classes,3)
    end_points={"output":net}

    return net,end_points




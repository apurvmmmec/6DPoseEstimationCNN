"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

import tensorflow as tf

from utils import *


def loss(logits, labels, classes):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height].
          The ground truth of your data.
      classes: numpy array - [classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    # with tf.name_scope('loss'):
    valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
        annotation_batch_tensor=tf.squeeze(labels, 3), logits_batch_tensor=logits, class_labels=classes)

    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                      labels=valid_labels_batch_tensor)
    cross_entropy_mean = tf.reduce_mean(softmax, name='cross-entropy')

    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    # return  cross_entropy_mean
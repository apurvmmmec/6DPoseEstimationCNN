from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import argparse
from skimage import io, transform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import fcn8_vgg as fcn8
import loss as cost
from dataGenerator import ImageDataGenerator
from inputLoader import *



base_path = '../../fcn8_vgg_data'
DATA_DIR = base_path+'/images/'
IMG_OUT_DIR = base_path+'/outputImages/'
IMG_IN_DIR = base_path+'/data/'
LOG_DIR = base_path+'/logs/'
MODEL = base_path+'/model/'
MODEL_NAME = 'fcn8_vgg'
DATASET = 'duck'
PRETRAIN_MODEL = base_path+'/pretrained/vgg16.npy'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('logDir', base_path+'./logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('checkpointPath', base_path+'./model',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                             """Whether to log device placement.""")

numTrainingExamples = 800;
# batch_size =4


MODE = False
TENSORBOARD = True
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
NUMBER_CHANNELS = 3

LEARNING_RATE = 1e-6
EPOCHS = 10000
BATCH_SIZE = 4
DISPLAY_STEP = 10
SAVE_STEP = 1000
EVALUATE_STE = 100

GPU = '0,1'

NUMBER_CLASSES = 2
IGNORE_LABEL = 0
KEEP_PROB = 0.5
TOWER_NAME = 'tower'



def parseArguments():
    parser = argparse.ArgumentParser(description='Train setting for FCN8')
    parser.add_argument('-m', '--mode', action='store_true', default=MODE, dest='mode',
                        help='True: Train model, False: Test model')
    parser.add_argument('-t''--tensorboard', action='store_true', default=TENSORBOARD, dest='tensorboard',
                        help='Tensorboard logging and visualization')
    parser.add_argument("-v", "--verbose", action="store", type=int, dest="verbose", default=0,
                        help="Verbosity level")
    parser.add_argument("-c", "--clean", action="store_true", dest="clean", default=True,
                        help="Clean and train from scratch")
    parser.add_argument("--eval-no-img-save", action="store_true", dest="evaluateStepDontSaveImages",
                        default=False, help="Don't save images on evaluate step")
    parser.add_argument("--gpu", action="store", dest="gpu",
                        default=GPU, help="Select GPU for training")

    parser.add_argument('--data-dir', action='store', default=DATA_DIR, dest='dataDir',
                        help='Root directory of dataset including train and test/validation')
    parser.add_argument('--split-data', action='store_true', default=False, dest='splitData',
                        help='Split data into train and test/validation')

    parser.add_argument('--dataset', action='store', default=DATASET,
                        choices=['PascalVOCContext', 'MITPlaces', 'PascalVOC', 'COCO'], dest='dataset',
                        help='Select dataset')
    parser.add_argument('--image-height', action='store', type=int, default=IMAGE_HEIGHT, dest='imageHeight',
                        help='Input image height feeding in network')
    parser.add_argument('--image-width', action='store', type=int, default=IMAGE_WIDTH, dest='imageWidth',
                        help='Input image width feeding in network')
    parser.add_argument("--image-channels", action="store", type=int, dest="imageChannels", default=NUMBER_CHANNELS,
                        help="Number of channels in image for feeding into the network")
    parser.add_argument("--random-fetch", action="store_true", dest="random", default=False,
                        help="Fetech random images for each batch")
    parser.add_argument('--input-text-present', action='store_true', default=False, dest='inputTextPresent',
                        help='Input text for loading images present')

    parser.add_argument('--learning-rate', action='store', type=float, default=LEARNING_RATE, dest='learningRate',
                        help='Learning rate of the model')
    parser.add_argument("--epochs", action="store", type=int, dest="trainingEpochs", default=EPOCHS,
                        help="Training epochs")
    parser.add_argument("--batchSize", action="store", type=int, dest="batchSize", default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--display-step", action="store", type=int, dest="displayStep", default=DISPLAY_STEP,
                        help="Progress display step")
    parser.add_argument("--save-step", action="store", type=int, dest="saveStep", default=SAVE_STEP,
                        help="Progress save step")
    parser.add_argument("--evaluate-step", action="store", type=int, dest="evaluateStep", default=EVALUATE_STE,
                        help="Progress evaluation step")

    parser.add_argument("--images-in-dir", action="store", dest="imagesInDir",
                        default=IMG_IN_DIR, help="Directory for list of input images and annotated labels")
    parser.add_argument("--pretrained-dir", action="store", dest="pretrained",
                        default=PRETRAIN_MODEL, help="Path to the pretrained model to load weights")
    parser.add_argument("--images-out-dir", action="store", dest="imagesOutDir",
                        default=IMG_OUT_DIR, help="Directory for saving output images")
    parser.add_argument("--log-dir", action="store", dest="logsDir", default=LOG_DIR,
                        help="Directory for saving logs")
    parser.add_argument("--model-dir", action="store", dest="modelDir", default=MODEL,
                        help="Directory for saving the model")
    parser.add_argument("--model-name", action="store", dest="modelName", default=MODEL_NAME,
                        help="Name to be used for saving the model")

    parser.add_argument("--num-classes", action="store", type=int, dest="numClasses", default=NUMBER_CLASSES,
                        help="Number of classes")
    parser.add_argument("--ignore-label", action="store", type=int, dest="ignoreLabel", default=IGNORE_LABEL,
                        help="Label to ignore for loss computation")
    parser.add_argument("--keep-prob", action="store", type=float, dest="keepProb", default=KEEP_PROB,
                        help="Probability of keeping a neuron active during training")

    return parser.parse_args()
def tower_loss(scope, images, labels):

  labelclasses = np.arange(2)
  labelclasses = np.append(labelclasses, [255])
  upscore32_pred = fcn8.inference(rgb=images)

  _ = cost.loss(upscore32_pred, labels, labelclasses)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(args):

  train_file_name = args.imagesInDir + 'train.txt'
  val_file_name = args.imagesInDir + 'val.txt'


  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(),tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(args.learningRate)

    # Get images and labels for CIFAR-10.
    trainDataGen = ImageDataGenerator(args, train_file_name,
                                      args.numClasses,
                                      'training',
                                      args.batchSize,
                                      num_preprocess_threads=8,
                                      shuffle=True,
                                      min_queue_examples=1000)

    images = trainDataGen.img_batch
    labels = trainDataGen.label_batch

    # images, labels = cifar10.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()

            loss = tower_loss(scope, image_batch, label_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    # summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.logDir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = args.batchSize * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.checkpointPath, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)



def testModel(args):
    print("Testing saved model")

    # os.system("rm -rf " + args.imagesOutDir)
    # os.system("mkdir " + args.imagesOutDir)
    test_file_name = args.imagesInDir + 'test.txt'
    print (test_file_name)

    print("Testing saved model")

    os.system("rm -rf " + args.imagesOutDir)
    os.system("mkdir " + args.imagesOutDir)

    testDataGen = ImageDataGenerator(args, test_file_name,
                                      args.numClasses,
                                      'test',
                                      1,
                                      num_preprocess_threads=1,
                                      shuffle=True,
                                      min_queue_examples=1)

    rgbImgBatch = testDataGen.img_batch
    labelBatch =  testDataGen.label_batch

    fcn8.inference(rgb=rgbImgBatch)

    # SAVER
    train_saver = tf.train.Saver()

    merged_summary = tf.summary.merge_all()
    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    train_saver = tf.train.import_meta_graph(args.modelDir + 'model.ckpt-51000' + ".meta")
    train_saver.restore(sess, args.modelDir + 'model.ckpt-51000')

    print('now testing {} files')
    tf.get_variable_scope().reuse_variables()

    epsilon = tf.constant(value=1e-4)


    upscore32_pred = fcn8.inference(rgb=rgbImgBatch)
    inputShape = rgbImgBatch.get_shape().as_list()
    inputShape[0] = -1  # self.batchSize # Images in batch
    inputShape[3] = args.numClasses
    logits = tf.reshape(upscore32_pred, (-1, args.numClasses))
    softmax = tf.nn.softmax(logits + epsilon)
    probabilities = tf.reshape(softmax, inputShape, name='probabilities')

    probabilities,labels = sess.run([probabilities,labelBatch])
    # labels = sess.run(labelBatch)
    testDataGen.saveImage(probabilities,labels );

    print('done calculating prb')

    print("Model tested!")
    summary_writer = tf.summary.FileWriter(FLAGS.logDir, sess.graph)
    summary_str = sess.run(merged_summary)
    summary_writer.add_summary(summary_str, 0)


def main(argv=None):  # pylint: disable=unused-argument

  os.system("rm -rf " + FLAGS.logDir)
  os.system("mkdir " + FLAGS.logDir)

  args = parseArguments()
  il = InputLoader(args)
  # testModel(args)
  train(args)


if __name__ == '__main__':
  tf.app.run()
# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
from os.path import join, isfile, splitext
import numpy as np
from scipy import stats
from skimage import io, transform

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, args,txt_file,num_classes, mode, batch_size, num_preprocess_threads=1, shuffle=True,
                 min_queue_examples=1):
        self.args =args
        self.txt_file = txt_file
        self.num_preprocess_threads = num_preprocess_threads
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.mode = mode
        self.imgShape = [self.args.imageHeight, self.args.imageWidth, self.args.imageChannels]
        self.maskShape = tf.stack([self.args.imageHeight, self.args.imageWidth])
        self.num_classes = int(num_classes)
        input_queue = tf.train.string_input_producer([txt_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line]).values

        if(mode=='training' or mode=='validation'):
            split_line = tf.string_split([line]).values


            rgb_image_path = split_line[0]
            label_image_path = split_line[1]


            self.image_o = self.read_image(rgb_image_path,0)
            # self.image = tf.subtract(self.image, VGG_MEAN)

            self.label_image_o = self.read_image(label_image_path,1)

            do_flip = tf.random_uniform([], 0, 1)
            self.image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(self.image_o), lambda: self.image_o)
            self.label_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(self.label_image_o), lambda: self.label_image_o)

            self.image.set_shape((self.args.imageHeight, self.args.imageWidth, 3))
            self.label_image.set_shape((self.args.imageHeight, self.args.imageWidth, 1))

            # self.img_batch, self.label_batch = tf.train.shuffle_batch([self.imageC, self.label],
            #                                 batch_size=batch_size,
            #                                 num_threads=num_preprocess_threads,
            #                                 capacity=min_queue_examples + 3 * batch_size,
            #                                 min_after_dequeue=min_queue_examples)

            self.img_batch, self.label_batch = tf.train.batch([self.image, self.label_image],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 3 * batch_size,
                                            )

        elif(mode=='test'):
            print'Generating test Image Batch'
            split_line = tf.string_split([line]).values

            rgb_image_path = split_line[0]
            self.image = self.read_image(rgb_image_path, 0)
            self.label = rgb_image_path;

            self.image.set_shape((self.args.imageHeight, self.args.imageWidth, 3))

            self.img_batch, self.label_batch = tf.train.batch([self.image,self.label],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 1 * batch_size,
                                                              )


    def string_length_tf(self,t):
        return tf.py_func(len, [t], [tf.int64])

    def read_image(self, image_path,mask):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = self.string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        if(mask==1):
            image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path), channels=1),
                        lambda: tf.image.decode_png(tf.read_file(image_path), channels=1))

            image = tf.image.resize_nearest_neighbor(tf.expand_dims(image, 0), self.maskShape)
            image = tf.squeeze(image, squeeze_dims=[0])
        else:
            image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path), channels=3),
                            lambda: tf.image.decode_png(tf.read_file(image_path), channels=3))
            image = tf.image.resize_images(image, [self.args.imageHeight, self.args.imageWidth], tf.image.ResizeMethod.AREA)

        image = tf.image.convert_image_dtype(image, tf.float32)


        return image

    def getAnnotationClasses(self):
        labelclasses = np.array([])
        # if self.args.dataset == 'mit':
        labelclasses = np.arange(2)
        labelclasses = np.append(labelclasses, [0])
        return labelclasses

    def saveImage(self,outDir,outputImages, filenames):
        numImages = len(filenames)

        def map_channels(i_x):
            i, x = i_x
            x = (x * 255).astype(np.uint8)
            if x.max() > 0.15 * 255:
                threshold = np.fabs(x.max() - x.max() * .85)
            else:
                threshold = 255
            threshImage = stats.threshold(x, threshmin=threshold)
            threshImage[threshImage > 0] = i
            return x

        def smash_channels(channels):
            base = channels[0]
            for i, x in enumerate(channels):
                base[x > 0] = i
            return base
        for i in range (0,numImages):

            # imgchannels = list(map(map_channels, enumerate(np.transpose(outputImages[i, :, :, :], [2, 0, 1]))))
            # smashed = smash_channels(imgchannels)

            print (outDir+'segout'+filenames[i][-8:])
            io.imsave(outDir+'segout'+filenames[i][-8:], (outputImages[i,:,:,1]*255).astype(np.uint8))
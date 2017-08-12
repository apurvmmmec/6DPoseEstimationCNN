# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf


VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file,data_path,num_classes, mode, batch_size, num_preprocess_threads=1, shuffle=True,
                 min_queue_examples=1):

        self.txt_file = txt_file
        self.num_preprocess_threads = num_preprocess_threads
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.mode = mode

        self.num_classes = int(num_classes)
        input_queue = tf.train.string_input_producer([txt_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line]).values

        if(mode=='training' or mode=='validation'):
            split_line = tf.string_split([line]).values

            if (mode == 'training'):
                rgb_image_path = tf.string_join([data_path + 'labelledRGB_patch/', split_line[0]])
                depth_image_path = tf.string_join([data_path + 'labelledDepth_patch/', split_line[0]])
            if (mode == 'validation'):
                rgb_image_path = tf.string_join([data_path + 'labelledRGB_patch/', split_line[0]])
                depth_image_path = tf.string_join([data_path + 'labelledDepth_patch/', split_line[0]])

            self.image = self.read_image(rgb_image_path)
            # self.image = tf.subtract(self.image, VGG_MEAN)
            self.image.set_shape((227, 227, 3))

            self.depth_image = self.read_image(depth_image_path)
            self.depth_image.set_shape((227, 227, 1))
            self.imageC = tf.concat([self.image, self.depth_image], 2)

            self.label = tf.string_to_number(split_line[1],tf.int32)
            # self.label = split_line[0]


            # self.img_batch, self.label_batch = tf.train.shuffle_batch([self.imageC, self.label],
            #                                 batch_size=batch_size,
            #                                 num_threads=num_preprocess_threads,
            #                                 capacity=min_queue_examples + 3 * batch_size,
            #                                 min_after_dequeue=min_queue_examples)

            self.img_batch, self.label_batch = tf.train.batch([self.imageC, self.label],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 3 * batch_size,
                                            )

        elif(mode=='test'):
            print'Generating test Image Batch'
            split_line = tf.string_split([line]).values
            rgb_image_path = tf.string_join([data_path + 'testPatchesRGB/', line])
            depth_image_path = tf.string_join([data_path + 'testPatchesDepth/', line])

            # rgb_image_path = tf.string_join([data_path + 'labelledRGB_patch/', line])
            # depth_image_path = tf.string_join([data_path + 'labelledDepth_patch/', line])

            # rgb_image_path = tf.string_join([data_path + 'labelledRGB_patch/', split_line[0]])
            # depth_image_path = tf.string_join([data_path + 'labelledDepth_patch/', split_line[0]])

            # depth_image_path =

            self.image = self.read_image(rgb_image_path)
            # self.image = tf.subtract(self.image, VGG_MEAN)
            self.image.set_shape((227, 227, 3))

            self.depth_image = self.read_image(depth_image_path)
            self.depth_image.set_shape((227, 227, 1))

            self.imageC = tf.concat([self.image, self.depth_image], 2)


            self.label = split_line[0]


            self.img_batch, self.label_batch = tf.train.shuffle_batch([self.imageC,self.label],
                                                  batch_size=batch_size,
                                                  num_threads=num_preprocess_threads,
                                                  capacity=min_queue_examples + 3 * batch_size,
                                                  min_after_dequeue=min_queue_examples)


    def string_length_tf(self,t):
        return tf.py_func(len, [t], [tf.int64])

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = self.string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [227, 227], tf.image.ResizeMethod.AREA)

        return image
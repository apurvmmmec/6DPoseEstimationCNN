# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Monodepth data loader.
"""

import tensorflow as tf


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, mode):
        self.data_path = data_path

        self.mode = mode
        self.left_image_batch = None
        self.right_image_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        if mode == 'test' and not self.params.do_stereo:
            left_image_path = tf.string_join([self.data_path, split_line[0]])
            left_image_o = self.read_image(left_image_path)
        else:
            rgb_image_path = tf.string_join(['./labelledRGB_patch/', split_line[0]])
            depth_image_path = tf.string_join(['./labelledRGB_patch/', split_line[0]])

            rgb_image = self.read_image(rgb_image_path)
            depth_image = self.read_image(depth_image_path)
            label = split_line[1]


        if mode == 'train':


            rgb_image.set_shape([None, None, 3])
            depth_image.set_shape([None, None, 1])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * 10
            self.rgb_image_batch, self.laell_batch = tf.train.shuffle_batch([rgb_image ,label],
                                                                                   10, capacity,
                                                                                   min_after_dequeue,
                                                                                   1)




    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))



        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [227, 227], tf.image.ResizeMethod.AREA)

        return image

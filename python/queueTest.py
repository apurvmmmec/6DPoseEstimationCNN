# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import convert_to_tensor
from dataGenerator import *
from alexnet import AlexNet
from datetime import datetime
import os
from testDataGen import *
from genColors import getColors

import argparse
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'



base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

# base_path= '../'
train_file = base_path+'test/05/train.txt'
validate_file = base_path+'test/05/validate.txt'
test_file =  base_path+'test/05/test.txt'





colors = getColors()

filewriter_path = "../logs/"
checkpoint_path = "../checkpoints/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

display_step=1

def parseArguments():
    parser = argparse.ArgumentParser(description='6D PoseEstimation')
    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--testImg', type=int, help='test image sequence no. ')
    return parser.parse_args();


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(args):
    batch_size = 1
    num_epochs = 100
    num_classes = 9
    learning_rate = 1e-4

    test_data_path = base_path + 'test/05/testImages/{:04d}/'
    test_file = base_path + 'test/05/testImages/{:04d}/test{:04d}_2020.txt'
    train_data_path = base_path + 'test/05/dummy/'
    validation_data_path = train_data_path

    if (args.mode=='train'):

        num_training_samples = count_text_lines(train_file)
        num_validation_samples = count_text_lines(validate_file)
        train_batches_per_epoch = np.ceil(num_training_samples / batch_size).astype(np.int32)
        val_batches_per_epoch = np.ceil(num_training_samples / batch_size).astype(np.int32)


        trainDataGen = ImageDataGenerator(train_file,
                                          train_data_path,
                                          num_classes,
                                          'training',
                                          batch_size,
                                          num_preprocess_threads=5,
                                          shuffle=True,
                                          min_queue_examples=1000)

        validationDataGen = ImageDataGenerator(validate_file,
                                               validation_data_path,
                                               num_classes,
                                               'validation',
                                               batch_size,
                                               num_preprocess_threads=2,
                                               shuffle=False,
                                               min_queue_examples=100)



        train_imgBatch = trainDataGen.img_batch
        train_labelBatch = trainDataGen.label_batch

        val_imgBatch = validationDataGen.img_batch
        val_labelBatch = validationDataGen.label_batch


        # TF placeholder for graph input and output
        x = tf.placeholder(tf.float32, [batch_size, 227, 227, 4])
        y = tf.placeholder(tf.float32, [batch_size, num_classes])

        train_layers = ['fc8']
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]


        # Initialize model
        model = AlexNet(x, num_classes,train_layers)

        score = model.fc8
        with tf.name_scope("cross_ent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))


        # Op for calculating the loss
        # Train op
        with tf.name_scope("train"):
            # Create optimizer and apply gradient descent to the trainable variables
            gOpt = tf.train.GradientDescentOptimizer(learning_rate)
            grads = gOpt.compute_gradients(loss)
            train_op = gOpt.apply_gradients(grads)


        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)


        # Add the loss to summary
        tf.summary.scalar('cross_entropy', loss)

        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)
        # Merge all summaries together
        merged_summary = tf.summary.merge_all()

    elif(args.mode=='test'):
        if (args.testImg == None):
            sys.exit('Please Input a valid image no with option --testImg NUMBER ')
        testImageIdx = args.testImg
        test_data_path = test_data_path.format(testImageIdx)
        test_file = test_file.format(testImageIdx,testImageIdx)
        num_test_samples = count_text_lines(test_file)
        batch_size = 1000
        test_batches_per_epoch = np.ceil(num_test_samples / batch_size).astype(np.int32)

        x = tf.placeholder(tf.float32, [batch_size, 227, 227, 4])
        # keep_prob = tf.placeholder(tf.float32)

        testDataGen = ImageDataGenerator(test_file,
                                         test_data_path,
                                         num_classes,
                                         'test',
                                         batch_size,
                                         num_preprocess_threads=5,
                                         shuffle=False,
                                         min_queue_examples=10000)
        test_imgBatch = testDataGen.img_batch
        test_labelBatch = testDataGen.label_batch

        # Initialize model
        model = AlexNet(x, num_classes,[])
        #
        # Link variable to model output
        score = model.fc8
        softmax = tf.nn.softmax(score)
        # testImg = getTestImage();
        outImg = np.zeros([480, 640,3])



    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(filewriter_path)


    with tf.Session() as sess:

        if(args.mode=='train'):
            # Required to get the filename matching to run.
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()


            model.load_initial_weights(sess)

            # saver = tf.train.import_meta_graph('../checkpoints/model_epoch100.ckpt.meta')
            # saver.restore(sess, "../checkpoints/model_epoch100.ckpt")

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)


            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              filewriter_path))

            for epoch in range(0,num_epochs):
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

                for step in range(0, train_batches_per_epoch):
                    print("{} Batch number: {}".format(datetime.now(), step + 1))



                    # Get an image tensor and print its value.
                    imageB, labelB = sess.run([train_imgBatch, train_labelBatch])
                    # #For Debugging
                    # np.save('imgDump.npy',imageB)
                    # np.save('labelDump.npy',labelB)

                    one_hot_labels = np.zeros((batch_size, num_classes))
                    for i in range(len(labelB)):
                        one_hot_labels[i][int(labelB[i])] = 1
                    #
                    # image_summary_t = tf.summary.image('t', tf.reshape(train_img, [1, 227, 227, 3]), max_outputs=1)
                    # _,_, image_summary = sess.run([train_img, train_label, image_summary_t])
                    # writer.add_summary(image_summary)
                    sess.run([train_op], feed_dict={x: imageB,
                                                     y: one_hot_labels})

                    # Generate summary with the current batch of data and write to file
                    if step % display_step == 0:
                        s = sess.run(merged_summary, feed_dict={x: imageB,
                                                     y: one_hot_labels})
                        writer.add_summary(s, epoch * train_batches_per_epoch + step)
                        # writer.add_summary(image_summary)

                    # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                test_acc = 0.
                test_count = 0
                for _ in range(val_batches_per_epoch):

                    imageB, labelB = sess.run([val_imgBatch, val_labelBatch])
                    one_hot_labels = np.zeros((batch_size, num_classes))
                    for i in range(len(labelB)):
                        one_hot_labels[i][int(labelB[i])] = 1

                    acc = sess.run(accuracy, feed_dict={x: imageB,
                                                    y: one_hot_labels,})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count

                print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))



                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

                    # # image_summary_t = tf.summary.image('v'+label_v, tf.reshape(val_img, [1, 227, 227, 3]), max_outputs=1)
                    # _,_,image_summary = sess.run([val_img, val_label, image_summary_t])
                    # writer.add_summary(image_summary)


            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)

        elif(args.mode=='test'):


            print 'inside Evaluation'
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            saver = tf.train.import_meta_graph('../checkpoints/model_epoch31.ckpt.meta')
            saver.restore(sess, "../checkpoints/model_epoch31.ckpt")
            print'Loaded Model from Checkpoint'

            # for step in range(0, test_batches_per_epoch):
            for step in range(0, test_batches_per_epoch):

                print("{} Batch number: {}".format(datetime.now(), step + 1))
                imageB ,labelB= sess.run([test_imgBatch, test_labelBatch])
                pixelLabels = labelB
                # print pixelLabels


                pred = sess.run(softmax, feed_dict={x: imageB})
                print pred

                pixelPred = np.argmax(pred,1)
                print pixelPred
                ct=0;
                for pixel in pixelLabels:
                    px = int(pixel[0:3])
                    py= int(pixel [3:6])
                    if((pixelPred[ct]==0)):
                        outImg[px, py, :] = 0
                    else:
                        outImg[px, py, 0] = (colors[pixelPred[ct] - 1, 2]) * 255
                        outImg[px, py, 1] = (colors[pixelPred[ct] - 1, 1]) * 255
                        outImg[px, py, 2] = (colors[pixelPred[ct] - 1, 0]) * 255
                    # if(pixelPred[ct]==0): # Background
                    #     outImg[px, py, :] = 255
                    # elif(pixelPred[ct]==1): # Foreground
                    #     outImg[px,py,2]=255
                    #     outImg[px, py, 1] = 255

                    ct=ct+1;

            # img = Image.fromarray(outImg)
            # img.show()
            cv2.imwrite('output_2.png%s'%(args.testImg), outImg)
            coord.request_stop()
            coord.join(threads)



def main(_):
    args = parseArguments()
    train(args)


if __name__ == '__main__':
    tf.app.run()
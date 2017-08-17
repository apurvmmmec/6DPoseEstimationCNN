import numpy as np
from tqdm import tqdm
import os
from os.path import join, isfile, splitext
import math
from skimage import io, transform
import cv2
from scipy import stats

from utils import *


class InputLoader():
    def __init__(self, args):

        self.args = args

        self.IMAGE = 'images/'
        self.ANNOTATION = 'annotations/'

        if not self.args.inputTextPresent:
            self._createDataTxt(imagePath=join(self.args.dataDir, self.args.dataset + '/', self.IMAGE),
                                annotationPath=join(self.args.dataDir, self.args.dataset + '/', self.ANNOTATION),
                                split=self.args.splitData)

        # self.imageListTrain = self._readFilenames(self.args.imagesInDir + 'train.txt')
        # self.imageListValid = self._readFilenames(self.args.imagesInDir + 'val.txt')
        #
        # if not self.args.random:
        #     np.random.shuffle(self.imageListTrain)
        #
        # self.currentIndex = 0
        # self.totalEpochs = 0
        # self.totalImages = len(self.imageListTrain)
        # self.totalImagesValid = len(self.imageListValid)
        #
        # self.imgShape = [self.args.imageHeight, self.args.imageWidth, self.args.imageChannels]
        # self.maskShape = tf.stack([self.args.ima # self.imageListTrain = self._readFilenames(self.args.imagesInDir + 'train.txt')
        # self.imageListValid = self._readFilenames(self.args.imagesInDir + 'val.txt')
        #
        # if not self.args.random:
        #     np.random.shuffle(self.imageListTrain)
        #
        # self.currentIndex = 0
        # self.totalEpochs = 0
        # self.totalImages = len(self.imageListTrain)
        # self.totalImagesValid = len(self.imageListValid)
        #
        # self.imgShape = [self.args.imageHeight, self.args.imageWidth, self.args.imageChannels]
        # self.maskShape = tf.stack([self.args.imageHeight, self.args.imageWidth])

    def _readFilenames(self, imageListFile):
        fileNames = []
        with open(imageListFile, 'r') as f:
            for line in tqdm(f, desc='Reading annotated files at: ' + imageListFile):
                fileNames.append(line.strip())
        return fileNames

    def _createDataTxt(self, imagePath, annotationPath, split=False):
        JPG = '.jpg'
        PNG ='.png'
        TRAINING = 'training/'
        # VALIDATION = 'validation/'

        if split:
            annotatedImages = os.listdir(annotationPath)
            # np.random.shuffle(annotatedImages)
            splitSize = math.ceil(len(annotatedImages) * 0.85)

            annotatedImagesTrain = annotatedImages[:splitSize]
            # annotatedImagesValidation = annotatedImages[splitSize:]
        else:
            annotatedImagesTrain = os.listdir(join(annotationPath, TRAINING))
            # annotatedImagesValidation = os.listdir(join(annotationPath, VALIDATION))

        with open(self.args.imagesInDir + 'train.txt', 'w') as file:
            for ann in tqdm(annotatedImagesTrain, desc='Writing train.txt for input dataset'):
                if isfile(join(imagePath, TRAINING, splitext(ann)[0]) + PNG):
                    file.write(' '.join(
                        [join(imagePath, TRAINING, splitext(ann)[0]) + PNG,
                         join(annotationPath, TRAINING, ann)]) + '\n')

        # with open(self.args.imagesInDir + 'val.txt', 'w') as file:
        #     for annv in tqdm(annotatedImagesValidation, desc='Writing valid.txt for input dataset'):
        #         if isfile(join(imagePath, VALIDATION, splitext(annv)[0]) + PNG):
        #             file.write(' '.join(
        #                 [join(imagePath, VALIDATION, splitext(annv)[0]) + PNG,
        #                  join(annotationPath, VALIDATION, annv)]) + '\n')

        return

    def readImagesFromDisk(self, fileNames, readMask=True):
        images = []
        mask = []
        for i in tqdm(range(0, len(fileNames)), desc='Reading images from disk'):
            # todo
            img = tf.read_file(fileNames[i].split(' ')[0])
            img = tf.image.decode_png(img, channels=3)

            # img = io.imread(fileNames[i].split(' ')[0])
            if img.shape != self.imgShape:
                img = tf.image.resize_images(img, self.maskShape)
                # img = transform.resize(img, self.imgShape, preserve_range=True)
            # skimage.io.imsave('./resizedIm/' + imageName, img)
            images.append(img)

            if readMask:
                label_contents = tf.read_file(fileNames[i].split(' ')[1])
                labeling = tf.image.decode_png(label_contents, channels=1)
                # img = io.imread(fileNames[i].split(' ')[1])
                if labeling.shape != self.imgShape:
                    labeling = tf.image.resize_nearest_neighbor(tf.expand_dims(labeling, 0), self.maskShape)
                    labeling = tf.squeeze(labeling, squeeze_dims=[0])
                    # labeling = transform.resize(labeling, self.imgShape, preserve_range=True)
                # skimage.io.imsave('./resizedIm/' + imageName, img)
                mask.append(labeling)

        return images, mask

    def getTrainBatch(self):
        if self.totalEpochs >= self.args.trainingEpochs:
            return None, None

        endIndex = self.currentIndex + self.args.batchSize
        if self.args.random:
            # Randomly fetch any images
            self.indices = np.random.choice(self.totalImages, self.args.batchSize)
        else:
            # Fetch the next sequence of images
            self.indices = np.arange(self.currentIndex, endIndex)

            if endIndex > self.totalImages:
                # Replace the indices which overshot with 0
                self.indices[self.indices >= self.totalImages] = np.arange(0, np.sum(self.indices >= self.totalImages))

        imageBatch, maskBatch = self.readImagesFromDisk([self.imageListTrain[index] for index in self.indices])

        self.currentIndex = endIndex
        if self.currentIndex > self.totalImages:
            print("Training epochs completed: %f" % (self.totalEpochs + (float(self.currentIndex) / self.totalImages)))
            self.currentIndex = self.currentIndex - self.totalImages
            self.totalEpochs = self.totalEpochs + 1

            # Shuffle the image list if not random sampling at each stage
            if not self.args.random:
                np.random.shuffle(self.imageListTrain)

        return tf.squeeze(tf.stack([imageBatch]), 0).eval(), tf.squeeze(tf.stack([maskBatch]), 0).eval()

    def getTestBatch(self, readMask=True):
        self.indices = np.random.choice(self.totalImagesValid, self.args.batchSize)
        imageBatch, maskBatch = self.readImagesFromDisk([self.imageListValid[index] for index in self.indices],
                                                        readMask=readMask)
        return tf.squeeze(tf.stack([imageBatch]), 0).eval(), tf.squeeze(tf.stack([maskBatch]), 0).eval()

    def saveLastBatchResults(self, outputImages, isTrain=True):
        if isTrain:
            imageNames = [self.imageListTrain[0]]# for index in self.indices]
        else:
            imageNames = [self.imageListValid[0]]# for index in self.indices]

        # Iterate over each image name and save the results
        for j in range(0, self.args.batchSize):
            imageName = imageNames[j].split('/')
            imageName = imageName[-1]
            if isTrain:
                imageName = self.args.imagesOutDir + '/' + 'train_' + imageName[:-4] + '_prob' + imageName[-4:]
            else:
                imageName = self.args.imagesOutDir + '/' + 'test_' + imageName[:-4] + '_prob' + imageName[-4:]

            def map_channels(i_x):
                i, x = i_x
                x = (x * 255).astype(np.uint8)
                if x.max() > 0.35 * 255:
                    threshold = np.fabs(x.max() - x.max() * .65)
                else:
                    threshold = 255
                threshImage = stats.threshold(x, threshmin=threshold)
                threshImage[threshImage > 0] = i
                return threshImage

            def smash_channels(channels):
                base = channels[0]
                for i,x in enumerate(channels):
                    base[x>0] = i
                return base


            imgchannels = list(map(map_channels, enumerate(np.transpose(outputImages[j, :, :, :], [2, 0, 1]))))
            smashed = smash_channels(imgchannels)
            # Save foreground probability
            #im = np.squeeze(outputImages[i, :, :, 1] * 255)
            #im = im.astype(np.uint8)  # Convert image from float to unit8 for saving
            io.imsave(imageName, smashed)

    def getAnnotationClasses(self):
        labelclasses = np.array([])
        if self.args.dataset == 'can':
            labelclasses = np.arange(2)
            labelclasses = np.append(labelclasses, [255])
        return labelclasses

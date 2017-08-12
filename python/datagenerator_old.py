import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot  as plt
import cv2

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/05/dummy/'
# base_path = './'
# rgb_numpy_array_path = base_path+'imageData/rgbImageArray{:03d}.npy'
# depth_numpy_array_path = base_path+'imageData/depthImageArray{:03d}.npy'
# label_numpy_array_path = base_path+'imageData/labelArray{:03d}.npy'

class ImageDataGenerator_Old:
    def __init__(self, class_list,mode, horizontal_flip=False, shuffle=False,
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227),
                 nb_classes = 2, npArrChunkSize=100):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        self.numpyArryChunkSize =npArrChunkSize
        self.rgbNpyArr = np.zeros([self.numpyArryChunkSize, 227, 227, 3])
        self.depthNpyArr = np.zeros([self.numpyArryChunkSize, 227, 227])
        self.labelNpyArr = np.zeros([self.numpyArryChunkSize])
        self.mode = mode #1 for training, 2 vor validation

        self.read_class_list(class_list)


        if  (mode==1):
            self.rgb_numpy_array_path = base_path + 'trainingData/rgbImageArray{:03d}.npy'
            self.depth_numpy_array_path = base_path + 'trainingData/depthImageArray{:03d}.npy'
            self.label_numpy_array_path = base_path + 'trainingData/labelArray{:03d}.npy'
        elif (mode ==2):
            self.rgb_numpy_array_path = base_path + 'validationData/rgbImageArray{:03d}.npy'
            self.depth_numpy_array_path = base_path + 'validationData/depthImageArray{:03d}.npy'
            self.label_numpy_array_path = base_path + 'validationData/labelArray{:03d}.npy'

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self,class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()

        # #store total number of data
        self.data_size = len(lines)
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """

        tempRGB = []
        tempDepth = []
        tempLabels = []

        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(self.rgbNpyArr.shape[0])
        for i in idx:
            tempRGB.append(self.rgbNpyArr[i, :, :, :])
            tempDepth.append(self.depthNpyArr[i,:,:])
            tempLabels.append(self.labelNpyArr[i])

        self.rgbNpyArr = np.array(tempRGB)
        self.depthNpyArr = np.array(tempDepth)
        self.labelNpyArr = np.array(tempLabels)

                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size,iter):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels

        # print 'Self Pointer %d'%(self.pointer)
        # iter = (self.pointer/batch_size) +1
        batchPerNumpyFile = self.numpyArryChunkSize / batch_size;
        # print 'Iteration %d'%(iter)
        if((iter -1)%batchPerNumpyFile ==0):
            numpyArrFileIdx=int( (iter -1)/batchPerNumpyFile)
            print 'Loading Numpy array file %s'%(self.rgb_numpy_array_path.format(numpyArrFileIdx))
            print 'Loading Numpy array file %s'%(self.depth_numpy_array_path.format(numpyArrFileIdx))
            print 'Loading Numpy array file %s'%(self.label_numpy_array_path.format(numpyArrFileIdx))

            self.rgbNpyArr = np.load(self.rgb_numpy_array_path.format(numpyArrFileIdx))
            self.depthNpyArr = np.load(self.depth_numpy_array_path.format(numpyArrFileIdx))
            self.labelNpyArr = np.load(self.label_numpy_array_path.format(numpyArrFileIdx))
            self.reset_pointer()



        labels = np.array(self.labelNpyArr[self.pointer : self.pointer+batch_size])
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])

        rgb = np.array(self.rgbNpyArr[self.pointer:self.pointer+batch_size,:,:,:])
        rgb = rgb.astype(np.float32)
        depth =np.array(self.depthNpyArr[self.pointer:self.pointer+batch_size,:,:])
        images[:,:,:,0:3] =rgb[...,::-1]
        images[:,:,:,3] =depth
        # cv2.imwrite('./test1.png',rgb[0,:,:,:])

        self.pointer += batch_size



        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][int(labels[i])] = 1

            #return array of images and labels
        return images, one_hot_labels

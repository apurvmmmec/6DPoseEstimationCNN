import numpy as np
import cv2
base_path = './'



rgb_numpy_array_path = base_path+'imageData/rgbImageArray{:03d}.npy'
depth_numpy_array_path = base_path+'imageData/depthImageArray{:03d}.npy'
label_numpy_array_path = base_path+'imageData/labelArray{:03d}.npy'
inputFile = base_path + 'validate.txt'
outFile = base_path + 'validate.txt'

with open(inputFile) as f:
    content = f.readlines()
lines = []
print content
for x in content:
    lines.append(x)

linesArr = np.array(lines)
print linesArr
np.random.shuffle(linesArr)
print linesArr
fout = open(outFile,'w')
for x in linesArr:
    fout.write(x)

#
# import numpy as np
# import cv2
# import time
# import matplotlib.pyplot  as plt
#
#
# MODE = 1  # 1 for train, 2 for validate
# PATCH_TYPE = 1 # 1 for RGB ,2 for depth
#
# base_path = './'
#
# rgbPatchPath = base_path+ 'labelledRGB_patch/'
# depthPatchPath =  base_path+ 'labelledDepth_patch/'
#
# if (MODE==1):# Training
#     inputFile = base_path+ 'train.txt'
# else: #Validation
#     inputFile = base_path+ 'validate.txt'
#
# # rgb_numpy_array_path = base_path+'TestrgbImageArray{:03d}.npy'
# # label_numpy_array_path =  base_path+'TestlabelArray{:03d}.npy'
# npArrChunkSize= 10000
#
# base_path = './'
#
# rgb_numpy_array_path = base_path+'imageData/rgbImageArray{:03d}.npy'
# depth_numpy_array_path = base_path+'imageData/depthImageArray{:03d}.npy'
# label_numpy_array_path = base_path+'imageData/labelArray{:03d}.npy'
#
# def saveRGBImageAsNumpyArray():
#     with open(inputFile) as f:
#         content = f.readlines()
#
#     images = []
#     for i in range(0,npArrChunkSize):
#         x=content[i]
#         x= x.strip()
#         name = x.split(' ')[0]
#         imgData = cv2.imread('%s/%s'%(rgbPatchPath,name))
#         images.append(imgData)
#
#     imgArr = np.array(images)
#     np.save(rgb_numpy_array_path.format(00),imgArr)
#
#
# def saveLabels():
#     with open(inputFile) as f:
#         content = f.readlines()
#     numFiles = len(content)
#
#     labels = []
#
#
#     for i in range(0,npArrChunkSize):
#         x=content[i]
#         x= x.strip()
#         label = x.split(' ')[1]
#         labels.append(label)
#
#     labelsArr = np.array(labels)
#     np.save(label_numpy_array_path.format(00),labelsArr)
#
#
#
#         # self.depthNpyArr = np.zeros([self.npArrChunkSize, 227, 227])
#         # self.labelNpyArr = np.zeros([self.npArrChunkSize])
# # saveRGBImageAsNumpyArray()
# # saveLabels();
# rgbNpyArr = np.load(rgb_numpy_array_path.format(0));
# labelArr = np.load(label_numpy_array_path.format(0));
#
# # im1 = rgbNpyArr[0,:,:,:]
# #
# # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# # cv2.imshow('image', im1)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # temprgb = np.zeros([npArrChunkSize, 227, 227, 3])
#
# # temprgb= np.copy(rgbNpyArr)
# # rgbNpyArr = np.zeros([npArrChunkSize, 227, 227, 3])
# start = time.time()
#
# # idx = np.random.permutation(rgbNpyArr.shape[0])
# # print idx
# # count=0;
# # tempRGB=[]
# # tempDepth=[]
# # tempLabels=[]
# # for i in idx:
# #     tempRGB.append(rgbNpyArr[i,:,:,:])
# #     tempLabels.append(labelArr[i])
# #
# # rgbNpyArr = np.array(tempRGB)
# # labelArr = np.array(tempLabels)
# # end = time.time()
# # print(end - start)
# # abc= temprgb[3,:,:,:]
# # np.random.shuffle(imageArr)
# # print(imageArr.shape)
# # imageArrNew = []
# #
# # idx = np.random.permutation(imageArr.shape[0])
# # for i in idx:
# #     imageArrNew.append(imageArr[i,:,:,:])
# #
# # imgArrNpy = np.array(imageArrNew)
# #
# im1 = rgbNpyArr[0,:,:,:]
# plt.imshow(rgbNpyArr[0,:,:,:]);
# plt.show()
#
# # print labelArr[0]
# # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# # cv2.imshow('image', im1)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

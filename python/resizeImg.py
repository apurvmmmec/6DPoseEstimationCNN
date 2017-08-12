import ruamel.yaml as yaml
import numpy as np
import inout as io
from PIL import Image
import tensorflow as tf
import cv2



images = []

with open('./train.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
# for i in range(0,21530):
#     x= content[i]
#     name = x.split(' ')[0]
#     # print name
#     path = './labelledDepth_patch/%s'%(name)
#     imgData = io.load_im(path)
#     images.append(imgData)
#     print imgData.shape
#     print len(images)
#
# imgArr = np.array(images)
#
# np.save('./imageDepthCombined.npy',imgArr)
# print imgArr.shape




# with open('./train.txt') as f:
#     content = f.readlines()
# for x in content:
#     x= x.strip()
#     name = x.split(' ')[0]
#     # print name
#     path = './labelledDepth_patch/%s'%(name)
#     imgData = io.load_im(path)
#     img = Image.fromarray(imgData)
#     # img.show()
#     imResized =img.resize([227,227])
#     # imResized.show()
#     imResized.save(path)
#     print path
img = cv2.imread('./labelledRGB_patch/000_0001_00001.png')
# cv2.imshow(img)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cv2.imshow('image', img)
cv2.resizeWindow('image', img.shape[0], img.shape[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
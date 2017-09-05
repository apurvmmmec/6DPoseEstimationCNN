import numpy as np
import cv2
from PIL import Image

imgArray = np.load('/Users/apurvnigam/study_ucl/MScThesis/6DPoseCNN/testImages/segoutUnmodified0007.png.npy')
# lblArray = np.load('labelDump.npy')
# print lblArray[4]
imgData = imgArray[:,:,1]
cv2.imshow('image', imgData)
cv2.resizeWindow('image', imgData.shape[0], imgData.shape[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
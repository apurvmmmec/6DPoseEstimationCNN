import numpy as np
import ruamel.yaml as yaml
from PIL import Image
import cv2
from random import randrange,uniform



base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
synthetic_images_path = base_path +'train/{:02d}/rgb/{:04d}.png'
rgbTransformed_images_path = base_path +'train/{:02d}/rgbTransformed/{:04d}.png'
seg_in_path = base_path + 'train/{:02d}/seg/{:04d}.png'
segTransformed_images_path = base_path +'train/{:02d}/segTransformed/{:04d}.png'
random_bg_path = base_path+'train/bg/bg{:05d}.png'
scene_gt_mpath = base_path + 'train/{:02d}/gt.yml'


numBgImages =111
objId =5
img_rows = 480
img_cols= 640


with open(scene_gt_mpath.format(objId), 'r') as f:
    gt_info = yaml.load(f, Loader=yaml.CLoader)


def transformAndSaveImage(src, srcMask, idx):
    # Select a random bg Image out of total numBGImages
    bgIndex = randrange(0, numBgImages - 1)
    ## Load and resize to 640x480
    dest = cv2.imread(random_bg_path.format(bgIndex))

    destMask = np.zeros([img_rows, img_cols]);
    dest = cv2.resize(dest, (img_cols, img_rows))

    # Apply Random scaling to Image
    scale = uniform(0.4, 0.65)
    src = cv2.resize(src, (0, 0), fx=scale, fy=scale)
    srcMask = cv2.resize(srcMask, (0, 0), fx=scale, fy=scale)
    newHeight, newWidth, _ = np.shape(src)

    # Apply Random Rotation to Image
    if (uniform(0, 1)>=0.5):
        rot = uniform(1, 45)
        M = cv2.getRotationMatrix2D((newWidth / 2, newHeight / 2), rot, 1)
        src = cv2.warpAffine(src, M, (newWidth, newHeight))
        srcMask = cv2.warpAffine(srcMask, M, (newWidth, newHeight))

    # Find a random location inside an image of original size where to paste the transformed image
    newStartRow = randrange(0, img_rows - newHeight)
    newStartCol = randrange(0, img_cols - newWidth)
    print newStartRow, newStartCol
    print newHeight, newWidth

    background = Image.fromarray(dest)
    foreground = Image.fromarray(src)
    foreground = foreground.convert('RGBA')
    background.paste(foreground, (newStartCol, newStartRow), foreground)

    # destMask[newStartRow:newStartRow+newHeight,newStartCol:newStartCol+newWidth] = srcMask
    destMaskImage = Image.fromarray(destMask);
    destMaskImage = destMaskImage.convert('1')

    srcMaskImage = Image.fromarray(srcMask);
    srcMaskImage = srcMaskImage.convert('1')

    # srcMaskImage=srcMaskImage.convert('RGBA')
    destMaskImage.paste(srcMaskImage, (newStartCol, newStartRow), srcMaskImage)
    # background.show()
    # destMaskImage.show()

    background.save(rgbTransformed_images_path.format(objId, idx));
    destMaskImage.save(segTransformed_images_path.format(objId, idx))


numImages = len(gt_info)
ct=0
for imId in range(0,numImages):
    print imId


    src = cv2.imread(synthetic_images_path.format(objId, imId), -1)
    srcMask = cv2.imread(seg_in_path.format(objId, imId), -1)


    # Save transformed Image
    transformAndSaveImage(src, srcMask,2*imId)

    if (uniform(0, 1)>=0.5):
        #Crop Image and then transform and save
        widthFactor = uniform(0.3, 0.7)
        heightFactor = uniform(0.3, 0.7)
        cent_R = img_rows/2
        cent_C = img_cols/2
        y =int(widthFactor*cent_C)
        x = int(heightFactor*cent_R)
        src = src[cent_R-x:cent_R+x,cent_C-y:cent_C+y :]
        srcMask = srcMask[cent_R-x:cent_R+x,cent_C-y:cent_C+y :]

    transformAndSaveImage(src, srcMask,(2*imId)+1)







import numpy as np
import cv2

MODE = 2  # 1 for train, 2 for validate
PATCH_TYPE = 1 # 1 for RGB ,2 for depth
NP_ARRAY_CHUNK_SIZE = 100

base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/05/dummy/'
# base_path='./'

# rgbPatchPath = base_path+ 'labelledRGB_patch_val'
# depthPatchPath =  base_path+ 'labelledDepth_patch_val'
#
# rgbPatchPathR = base_path+ 'labelledRGB_patch_valR'
# depthPatchPathR =  base_path+ 'labelledDepth_patch_valR'

if (MODE==1):# Training
    inputFile = base_path+ 'trainingData/train.txt'
    rgb_numpy_array_path = base_path + 'trainingData/rgbImageArray{:03d}.npy'
    depth_numpy_array_path = base_path + 'trainingData/depthImageArray{:03d}.npy'
    label_numpy_array_path = base_path + 'trainingData/labelArray{:03d}.npy'

    rgbPatchPath = base_path + 'labelledRGB_patch'
    depthPatchPath = base_path + 'labelledDepth_patch'

else: #Validation
    inputFile = base_path+ 'validationData/validate.txt'
    rgb_numpy_array_path = base_path + 'validationData/rgbImageArray{:03d}.npy'
    depth_numpy_array_path = base_path + 'validationData/depthImageArray{:03d}.npy'
    label_numpy_array_path = base_path + 'validationData/labelArray{:03d}.npy'
    rgbPatchPath = base_path + 'labelledRGB_patch_val'
    depthPatchPath = base_path + 'labelledDepth_patch_val'


def resizeImages():
    with open(inputFile) as f:
        content = f.readlines()

    for x in content:
        x= x.strip()
        name = x.split(' ')[0]
        path = '%s/%s'%(rgbPatchPath, name)
        img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        imResized =cv2.resize(img,(227,227))
        resizePath = '%s/%s'%(rgbPatchPathR, name)
        cv2.imwrite(resizePath,imResized)
        print path

def saveLabels():
    with open(inputFile) as f:
        content = f.readlines()
    numFiles = len(content)
    print 'Num of training files %d'%(numFiles)
    numChunks = np.floor(numFiles / NP_ARRAY_CHUNK_SIZE).astype(np.int16)
    print 'Num of Label Files Being created %d'%(numChunks)
    len(content)
    for n in range(0,numChunks):
        print 'Creating %d Label File %s'%(n , label_numpy_array_path.format(n))
        labels = []
        startIdx = (n)*NP_ARRAY_CHUNK_SIZE
        endIdx = startIdx+ NP_ARRAY_CHUNK_SIZE

        for i in range(startIdx,endIdx):
            x=content[i]
            x= x.strip()
            label = x.split(' ')[1]
            labels.append(label)

        labelsArr = np.array(labels)
        np.save(label_numpy_array_path.format(n),labelsArr)


def saveRGBImageAsNumpyArray():
    with open(inputFile) as f:
        content = f.readlines()
    numFiles = len(content)
    print 'Num of training files %d'%(numFiles)

    numChunks = np.floor(numFiles / NP_ARRAY_CHUNK_SIZE).astype(np.int16)
    print 'Num of Label Files Being created %d'%(numChunks)

    len(content)
    for n in range(0,numChunks):
        print 'Creating %d RGBImages Numpy File %s'%(n , rgb_numpy_array_path.format(n))

        images = []
        startIdx = (n)*NP_ARRAY_CHUNK_SIZE
        endIdx = startIdx+ NP_ARRAY_CHUNK_SIZE

        for i in range(startIdx,endIdx):
            x=content[i]
            x= x.strip()
            name = x.split(' ')[0]
            imgData = cv2.imread('%s/%s'%(rgbPatchPath,name))
            images.append(imgData)


        imgArr = np.array(images)
        print 'Saving %s' % (rgb_numpy_array_path.format(n))
        np.save(rgb_numpy_array_path.format(n),imgArr)

def saveDepthImagesAsNumpyArray():
    with open(inputFile) as f:
        content = f.readlines()
    numFiles = len(content)
    numChunks = np.floor(numFiles / NP_ARRAY_CHUNK_SIZE).astype(np.int16)
    len(content)
    for n in range(0,numChunks):
        images = []
        startIdx = (n)*NP_ARRAY_CHUNK_SIZE
        endIdx = startIdx+ NP_ARRAY_CHUNK_SIZE

        for i in range(startIdx,endIdx):
            x=content[i]
            x= x.strip()
            name = x.split(' ')[0]
            imgData = cv2.imread('%s/%s'%(depthPatchPath,name),-1)
            imgData = imgData.astype(np.float32)
            images.append(imgData)

        imgArr = np.array(images)
        print 'Saving %s' % (depth_numpy_array_path.format(n))
        np.save(depth_numpy_array_path.format(n),imgArr)




# resizeImages()

saveRGBImageAsNumpyArray()
# saveDepthImagesAsNumpyArray()
saveLabels()

# imageArr = np.load(depth_numpy_array_path.format(0));
# print(imageArr.shape)
# im1 = imageArr[1,:,:]
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', im1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

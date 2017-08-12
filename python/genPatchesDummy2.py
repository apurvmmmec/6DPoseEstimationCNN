import numpy as np
import cv2
import time
import ruamel.yaml as yaml
import os
import shutil

from getTransf import getTransf

MODE = 1;  # 1 training #2 validation
# base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
base_path = '../'
rgb_in_mpath = base_path + 'test/{:02d}/rgb/{:04d}.png'
depth_in_mpath = base_path + 'test/{:02d}/depth/{:04d}.png'
seg_in_path = base_path + 'test/{:02d}/seg/{:04d}.png'
labelled_color_mpath = base_path + 'test/{:02d}/8bins/labelled_colored/{:04d}.png'
labelled_mpath = base_path + 'test/{:02d}/8bins/labelled/{:04d}.png'

train_file_path = base_path + 'test/{:02d}/train.txt'
val_file_path = base_path + 'test/{:02d}/validate.txt'

# train_file_path =  './train.txt'
# val_file_path = './validate.txt'

if (MODE == 1):
    if os.path.isdir(base_path + 'test/05/dummy/labelledRGB_patch'):
        print('Deleting %s' % (base_path + 'test/05/dummy/labelledRGB_patch'))
        shutil.rmtree(base_path + 'test/05/dummy/labelledRGB_patch')
        os.mkdir(base_path + 'test/05/dummy/labelledRGB_patch')
    else:
        os.mkdir(base_path + 'test/05/dummy/labelledRGB_patch')

    if os.path.isdir(base_path + 'test/05/dummy/labelledDepth_patch'):
        print('Deleting %s' % (base_path + 'test/05/dummy/labelledDepth_patch'))
        shutil.rmtree(base_path + 'test/05/dummy/labelledDepth_patch')
        os.mkdir(base_path + 'test/05/dummy/labelledDepth_patch')
    else:
        os.mkdir(base_path + 'test/05/dummy/labelledDepth_patch')

    labelledRGB_patch_path = base_path + 'test/{:02d}/dummy/labelledRGB_patch/{:03d}{:03d}_{:04d}_{:03d}.png'
    labelledDepth_patch_path = base_path + 'test/{:02d}/dummy/labelledDepth_patch/{:03d}{:03d}_{:04d}_{:03d}.png'

# else:
#     labelledRGB_patch_path = base_path + 'test/{:02d}/dummy/labelledRGB_patch_val/{:03d}_{:04d}_{:05d}.png'
#     labelledDepth_patch_path = base_path + 'test/{:02d}/dummy/labelledDepth_patch_val/{:03d}_{:04d}_{:05d}.png'

# train_labels_path = base_path+'test/{:02d}/train.txt'
# validation_labels_path = base_path+'test/{:02d}/validate.txt'
# test_labels_path = base_path+'test/{:02d}/test.txt'


seg_in_mpath = base_path + 'test/{:02d}/seg/{:04d}.png'
model_mpath = base_path + 'models/obj_{:02d}.ply'  # Already transformed

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'

labelled_images = []
all_bg_images = []
shuffled_labelled_images = []
train_images = []
val_images = []

save_patches = 1;


def getRotTrans(im_id, objId):
    # for im_id, gts_im in gts.items():
    gts_im = gts[im_id]
    for gt in gts_im:
        if 'obj_id' in gt.keys():
            if (gt['obj_id'] == objId):

                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                    r = gt['cam_R_m2c']
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
                    t = gt['cam_t_m2c']

    return r, t


# def getRandomBGImages(num):
#     numTotalBGFiles = len(all_bg_images)
#     perms = np.random.permutation(numTotalBGFiles)
#     idx = perms[:num]
#     for i in idx:
#         labelled_images.append(all_bg_images[i])


def shuffleFile():
    linesArr = np.array(labelled_images)
    np.random.shuffle(linesArr)

    shuffled_labellled_images = []
    for x in linesArr:
        shuffled_labellled_images.append(x)
    return shuffled_labellled_images;


def genTrainingValidation():
    numAllFiles = len(shuffled_labelled_images)
    perms = np.random.permutation(numAllFiles)
    numTrain = int(np.floor(numAllFiles * 0.9))
    idx = perms[:numTrain]
    for i in idx:
        train_images.append(shuffled_labelled_images[i])
    idx = perms[numTrain:]
    for i in idx:
        val_images.append(shuffled_labelled_images[i])
    return train_images, val_images


def subSampleImages(n, w, h, cenR, cenC):
    # n=1
    print n
    # start = time.time()


    rgbImg = cv2.imread(rgb_in_mpath.format(objId, imId), -1)
    depImg = cv2.imread(depth_in_mpath.format(objId, imId), -1)

    segImg = cv2.imread(seg_in_mpath.format(objId, imId), -1)

    labelImg = cv2.imread(labelled_mpath.format(objId, imId), -1)

    validPixels = []
    bgPixels = []
    # getPixels in segmentation mask
    for r in range(0 + h, 480 - h):
        for c in range(0 + w, 640 - w):
            if (segImg[r, c] > 0):  # Pixels Lying on Object
                if (labelImg[r, c] > 0):
                    validPixels.append([r, c])
            else:
                bgPixels.append([r, c])

    samplesToUse = []
    bgToUse = []

    numValidPixels = len(validPixels)
    perms = np.random.permutation(numValidPixels)
    idx = perms[:1000]
    for i in idx:
        samplesToUse.append(validPixels[i])

    numBGPixels = len(bgPixels)
    perms = np.random.permutation(numBGPixels)
    idx = perms[:170]
    for i in idx:
        bgToUse.append(bgPixels[i])

    ct = 0;
    for px in samplesToUse:
        # print px[0], px[1]
        r = px[0]
        c = px[1]
        label = int(labelImg[r, c])
        # print label
        rgbSubImg = rgbImg[r - h:r + h, c - w:c + w];
        depSubImg = depImg[r - h:r + h, c - w:c + w];
        if (save_patches == 1):
            cv2.imwrite(labelledRGB_patch_path.format(objId, r, c, n, label), rgbSubImg)
            cv2.imwrite(labelledDepth_patch_path.format(objId, r, c, n, label), depSubImg)

        labelled_images.append('%03d%03d_%04d_%03d.png %s\n' % (r, c, n, label, label))
        if ((label - 1) < 0):
            print (label - 1)
        lblCnt[label - 1, 0] = lblCnt[label - 1, 0] + 1
        ct += 1

    for px in bgToUse:
        # print px[0], px[1]
        r = px[0]
        c = px[1]
        rgbSubImg = rgbImg[r - h:r + h, c - w:c + w];
        depSubImg = depImg[r - h:r + h, c - w:c + w];
        if (save_patches == 1):
            rgbSubImg = cv2.resize(rgbSubImg, (227, 227))
            depSubImg = cv2.resize(depSubImg, (227, 227))

            cv2.imwrite(labelledRGB_patch_path.format(objId, r, c, n, 0), rgbSubImg)
            cv2.imwrite(labelledDepth_patch_path.format(objId, r, c, n, 0), depSubImg)

        labelled_images.append('%03d%03d_%04d_%03d.png %s\n' % (r, c, n, 0, 0))
        lblCnt[8, 0] = lblCnt[8, 0] + 1
        ct += 1

    return lblCnt


start = 0
scene_id = 5
objId = 5;
rotY180 = np.array([-1, 0, 0,
                    0, 1, 0,
                    0, 0, -1]).reshape([3, 3]);

rotZ180 = np.array([-1, 0, 0,
                    0, -1, 0,
                    0, 0, 1]).reshape([3, 3]);

K = np.zeros((3, 3));

K[0, 0] = 572.41140
K[0, 2] = 325.2611
K[1, 1] = 573.57043
K[1, 2] = 242.04899
K[2, 2] = 1

with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gts = yaml.load(f, Loader=yaml.CLoader)

rgbImg = cv2.imread(rgb_in_mpath.format(objId, start), -1)
depImg = cv2.imread(depth_in_mpath.format(objId, start), -1)

[r, t] = getRotTrans(start, objId)
# print r
# print t
r = r.dot(rotY180)
r = r.dot(rotZ180)
modelCen_h = np.array([0, 0, 0, 1]).reshape(4, 1);
P = K.dot(np.hstack((r, t)))
pts_im = P.dot(modelCen_h)
pts_im /= pts_im[2, :]

cenC0 = int(pts_im[0, 0])
cenR0 = int(pts_im[1, 0])
dep0 = depImg[cenR0, cenC0];
w0 = 20;
h0 = 20;
subImg = rgbImg[cenR0 - h0:cenR0 + h0, cenC0 - w0:cenC0 + w0, :];

lblCnt = np.zeros((9, 1), dtype=np.uint32);
testImagesCount = 0

for imId in range(1, 20, 1):
    if (imId % 10 != 0):
        rgbImg = cv2.imread(rgb_in_mpath.format(objId, imId), -1)
        depImg = cv2.imread(depth_in_mpath.format(objId, imId), -1)

        [r, t] = getRotTrans(imId, objId)
        r = r.dot(rotY180)
        r = r.dot(rotZ180)
        modelCen_h = np.array([0, 0, 0, 1]).reshape(4, 1);
        P = K.dot(np.hstack((r, t)))
        pts_im = P.dot(modelCen_h)
        pts_im /= pts_im[2, :]

        cenC = int(pts_im[0, 0])
        cenR = int(pts_im[1, 0])

        dep = depImg[cenR - 1, cenC - 1];
        if (dep == 0):
            print'depth is invalid'
            continue

        h1 = int(np.round((h0 * dep0) / float(dep)));
        w1 = int(np.round((w0 * dep0) / float(dep)));

        subSampleImages(imId, w1, h1, cenR, cenC);
    else:
        testImagesCount += 1;

labelled_patch_ct = sum(lblCnt)
bg_patch_count = labelled_patch_ct
# bg_images = getRandomBGImages(int(bg_patch_count))

shuffled_labelled_images = shuffleFile()

print lblCnt

trainImages, valImages = genTrainingValidation()
print len(train_images)
print  len(val_images)

trainFile = open(train_file_path.format(objId), 'w')
valFile = open(val_file_path.format(objId), 'w')

numTrainImages = len(train_images)
for i in range(0, numTrainImages):
    trainFile.write(train_images[i])
numValImages = len(val_images)
for i in range(0, numValImages):
    valFile.write(val_images[i])

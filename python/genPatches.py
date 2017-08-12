import numpy as np
import cv2
import time
import inout as io

import ruamel.yaml as yaml

from PIL import Image
from getTransf import getTransf


base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'test/{:02d}/rgb/{:04d}.png'
depth_in_mpath = base_path+ 'test/{:02d}/depth/{:04d}.png'
labelled_color_mpath = base_path+ 'test/{:02d}/labelled_colored/{:04d}.png'
labelled_mpath = base_path+ 'test/{:02d}/labelled/{:04d}.png'
labelledRGB_patch_path = base_path+'test/{:02d}/labelledRGB_patch/{:03d}_{:04d}_{:05d}.png'
labelledDepth_patch_path = base_path+'test/{:02d}/labelledDepth_patch/{:03d}_{:04d}_{:05d}.png'

labelledRGB_patch_path_val = base_path+'test/{:02d}/labelledRGB_patch_val/{:03d}_{:04d}_{:05d}.png'
labelledDepth_patch_path_val = base_path+'test/{:02d}/labelledDepth_patch_val/{:03d}_{:04d}_{:05d}.png'

train_labels_path = base_path+'test/{:02d}/train.txt'
validation_labels_path = base_path+'test/{:02d}/validate.txt'
test_labels_path = base_path+'test/{:02d}/test.txt'



seg_in_mpath = base_path+ 'test/{:02d}/seg/{:04d}.png'

model_mpath = base_path + 'models/obj_{:02d}.ply'  # Already transformed

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'



def getRotTrans(im_id,objId):

    # for im_id, gts_im in gts.items():
    gts_im = gts[im_id]
    for gt in gts_im:
        if 'obj_id' in gt.keys():
            if (gt['obj_id'] ==objId):

                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                    r= gt['cam_R_m2c']
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
                    t= gt['cam_t_m2c']

    return r,t


def subSampleImages(n,w,h,cenR,cenC):
    # n=1
    print n
    start = time.time()
    str = './RGB-D/rgb_noseg/color_%05d.png'%(n)
    rgbImg = io.load_im(rgb_in_mpath.format(objId, n))
    depImg = io.load_im(depth_in_mpath.format(objId, n))
    labelImg = io.load_im(labelled_mpath.format(objId, n))

    # rgbImg = cv2.imread(rgb_in_mpath.format(objID,n),cv2.IMREAD_UNCHANGED)
    # labelImg = cv2.imread('./objCordLabel/Can/label_%05d.png'%(n),cv2.IMREAD_UNCHANGED)
    # depImg = cv2.imread('./RGB-D/depth_noseg/depth_%05d.png'%(n),cv2.IMREAD_UNCHANGED)



    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', depImg)
    # cv2.resizeWindow('image', depImg.shape[0], depImg.shape[1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # w = 40;
    # h = 59;
    # cenR= 262
    # cenC= 373
    width = 2 * w;
    height = 2 * h;
    (rows,cols,channels)= rgbImg.shape
    # print(a)

    ct = 0;
    for r in range(0,rows - height,4):
        for c in range(0,cols-width,4):
            patchCenR = r + h - 1;
            patchCenC = c + w - 1;
            ct = ct + 1;
            label = int(labelImg[r + h - 1, c + w - 1])
            rgbSubImg = rgbImg[r:r + height - 1, c:c + width - 1];
            depSubImg = depImg[r:r + height - 1, c:c + width - 1];


            if ( (patchCenR >= (cenR - h-20)) and (patchCenR <= (cenR + h+20)) and (patchCenC >= (cenC - w-20)) and (patchCenC <= (cenC + w+20))):
                if (label > 0):

                    # rgbPatch = Image.fromarray(rgbSubImg, 'RGB')
                    # rgbPatch.save(labelledRGB_patch_path.format(objId,label,n,ct))
                    #
                    # depthPatch = Image.fromarray(depSubImg)
                    # depthPatch.save(labelledDepth_patch_path.format(objId,label,n, ct))
                    # trainInput.write('%03d_%04d_%05d.png %s\n' % (label, n, ct, label))

                    lblCnt[label-1,0]=lblCnt[label-1,0]+1

                else:
                    if(((r+1)%10==0) and ((c+1)%10==0)):

                        # rgbPatch = Image.fromarray(rgbSubImg, 'RGB')
                        # rgbPatch.save(labelledRGB_patch_path.format(objId,label,n, ct))
                        # #
                        # depthPatch = Image.fromarray(depSubImg)
                        # depthPatch.save(labelledDepth_patch_path.format(objId,label,n, ct))
                        trainInput.write('%03d_%04d_%05d.png %s\n' % (label, n, ct, label))

                        lblCnt[64, 0] = lblCnt[64, 0] + 1


            else:

                if ((r % 160 == 0) and (c % 160 == 0)):

                    # rgbPatch = Image.fromarray(rgbSubImg, 'RGB')
                    # rgbPatch.save(labelledRGB_patch_path.format(objId,label,n, ct))
                    #
                    # depthPatch = Image.fromarray(depSubImg)
                    # depthPatch.save(labelledDepth_patch_path.format(objId,label,n, ct))
                    # trainInput.write('%03d_%04d_%05d.png %s\n' % (label, n, ct, label))

                    lblCnt[64, 0] = lblCnt[64, 0] + 1

    end = time.time()
    print(end - start)
    return lblCnt



start=0
scene_id=5
objId =5;
rotY180=np.array([-1   ,  0  ,   0,
     0 ,    1 ,    0,
     0 ,    0 ,   -1]).reshape([3,3]);

rotZ180=np.array([-1   ,  0  ,   0,
     0 ,    -1 ,    0,
     0 ,    0 ,   1]).reshape([3,3]);

K = np.zeros((3, 3));

K[0, 0] = 572.41140
K[0, 2] = 325.2611
K[1, 1] = 573.57043
K[1, 2] = 242.04899
K[2, 2] = 1


with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gts = yaml.load(f, Loader=yaml.CLoader)

rgbImg = io.load_im(rgb_in_mpath.format(objId,start))
depImg = io.load_im(depth_in_mpath.format(objId,start))

[r,t]= getRotTrans(start,objId)
# print r
# print t
r=r.dot(rotY180)
r=r.dot(rotZ180)
modelCen_h = np.array([0,0,0,1]).reshape(4,1);
P = K.dot(np.hstack((r, t)))
pts_im = P.dot(modelCen_h)
pts_im /= pts_im[2, :]

cenC0 = int(pts_im[0,0])
cenR0 = int(pts_im[1,0])
dep0 = depImg[cenR0,cenC0];
w0=50;
h0=50;
subImg = rgbImg[cenR0-h0:cenR0+h0,cenC0-w0:cenC0+w0,:];
# img = Image.fromarray(subImg,'RGB')
#
# img.show();
# img.save(seg_path.format(objId,im_id))
# img.close()

# img.show()
    # im.save('fig1_modified.png')
    # img = Image.fromarray(m_rgb)
    # img = img.convert('1')
# toc,linesRead=getTransf(start);
# # print(toc)
# # print(np.array([0,0,0,1]).transpose())
# camCord = np.matmul(toc,np.array([0,0,0,1]))
#
# xc = camCord[0];
# yc = camCord[1];
# zc = camCord[2];
# #
# d  =- zc;
# cenC0 = int(np.round( ((xc*(572.41140)/d)+325.2611)));
# cenR0 = int(np.round((-(yc*(573.57043)/d)+242.04899)));

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# cv2.imshow('image', subImg)
# cv2.resizeWindow('image', subImg.shape[0], subImg.shape[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
lblCnt = np.zeros((65, 1),dtype=np.uint32);
trainInput = open(train_labels_path.format(objId), 'w')
# validationInput = open(validation_labels_path.format(objId), 'w')
# testInput = open(test_labels_path.format(objId),'w')

for imId in range(1,1190,10):
    rgbImg = io.load_im(rgb_in_mpath.format(objId, imId))
    depImg = io.load_im(depth_in_mpath.format(objId, imId))

    # toc1,linesRead = getTransf(n);

    # if(linesRead != 4):
    #     print 'Empty '
    #     continue
    #
    # camCord = np.matmul(toc1, np.array([0, 0, 0, 1]))
    #
    # xc = camCord[0];
    # yc = camCord[1];
    # zc = camCord[2];
    #
    # d = - zc;
    # cenC = int(np.round(((xc * (572.41140) / d) + 325.2611)));
    # cenR = int(np.round((-(yc * (573.57043) / d) + 242.04899)));

    [r, t] = getRotTrans(imId, objId)
    # print r
    # print t
    r = r.dot(rotY180)
    r = r.dot(rotZ180)
    modelCen_h = np.array([0, 0, 0, 1]).reshape(4, 1);
    P = K.dot(np.hstack((r, t)))
    pts_im = P.dot(modelCen_h)
    pts_im /= pts_im[2, :]

    cenC = int(pts_im[0, 0])
    cenR = int(pts_im[1, 0])

    dep = depImg[cenR-1, cenC-1];
    if(dep ==0):
        print'depth is invalid'
        continue

    h1 = int(np.round((h0 * dep0) / float(dep)));
    w1 = int(np.round((w0 * dep0) / float(dep)));
    # subImg = rgbImg[cenR - h1:cenR + h1, cenC - w1:cenC + w1, :];
    # img = Image.fromarray(subImg, 'RGB')
    # # #
    # img.show();
    # # img.save(seg_path.format(objId,im_id))
    # img.close()

    subSampleImages(imId, w1, h1, cenR, cenC);

print (lblCnt)
print sum(lblCnt)
#
#
# subSampleImages(1,40,59,262,373);
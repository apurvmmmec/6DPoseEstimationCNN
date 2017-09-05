import ruamel.yaml as yaml
import numpy as np
import inout as io
from PIL import Image
import renderer
# import matplotlib.pyplot as plt
import os
import time
import cv2

from utils import *
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'test/{:02d}/rgb/{:04d}.png'
depth_in_path = base_path+'test/{:02d}/depth/{:04d}.png'
model_mpath = base_path + 'models/obj_{:02d}.ply' # Already transformed
seg_path = base_path+'test/{:02d}/seg/'


centre_obj_id = 2;
objId =scene_id =2;


if (not(os.path.isdir(seg_path.format(objId)))):
    print("Making Segmentation Directory")
    os.mkdir(seg_path.format(objId))
# bbox_cens_path = 'output/bbox_cens.yml'

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'


w, h = 640, 480



depthMask = np.zeros([480,640])


with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gt_info = yaml.load(f, Loader=yaml.CLoader)

models=[]
for x in range(1,16):
    models.append(io.load_ply(model_mpath.format(x)))

numImages = len(gt_info)


def createMask(x):
    objId = gt_info_im[x]['obj_id']

    [r, t] = getRotTrans(gt_info, im_id, objId)

    objDepImg = renderer.render(models[objId - 1], (w, h), cam_mat, r, t, shading='phong', mode='depth')
    time.sleep(1)
    g1 = 0  # Original value
    g2 = 100000  # Value that we want to replace it with

    gray = objDepImg[:, :]
    mask = (gray == g1)
    objDepImg[:, :][mask] = [g2]
    oid = list(map(lambda x: 0. if x == 100000. else 0 if objId!=5  else 255, objDepImg.flatten()))

    return zip(objDepImg.flatten(), oid)


def pixelOverwrite(p1, p2):
    if p1[0] < p2[0]:
        return p1
    else:
        return p2


def mergeMasks(a, b):
    return list(map(pixelOverwrite, a, b))



for im_id in range(0,numImages):

    rgb = io.load_im(rgb_in_mpath.format(centre_obj_id,im_id))
    depth = io.load_im(rgb_in_mpath.format(centre_obj_id,im_id))
    depthMask = np.ones([480,640])*100000;

    gt_info_im = gt_info[im_id]
    numObjs = len(gt_info_im)

    objList = np.arange(0,numObjs)
    print objList



    start = time.time()
    depthMaskedImage = list(map(createMask, objList))
    mergedDepthMaskedImage = [zip(depthMask.flatten(),np.zeros(480*640))]+ depthMaskedImage
    maskedImage = reduce(mergeMasks,depthMaskedImage)

    dt = np.dtype('float','float')
    masked = np.array(maskedImage,dtype=dt)
    fileName = seg_path.format(centre_obj_id) + '%04d.png' % (im_id)
    cv2.imwrite(fileName,np.reshape(masked[:, 1], [480, 640]));
    end = time.time()
    print (end -start)


    print('ok')
import ruamel.yaml as yaml
import numpy as np
import inout as io
from PIL import Image
import renderer
import os
import time

from utils import *
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'train/{:02d}/rgb/{:04d}.png'
model_mpath = base_path + 'models/obj_{:02d}.ply' # Already transformed



objId =5
scene_id =5;



# bbox_cens_path = 'output/bbox_cens.yml'

scene_info_mpath = base_path + 'train/{:02d}/info.yml'
scene_gt_mpath = base_path + 'train/{:02d}/gt.yml'


w, h = 640, 480





with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gt_info = yaml.load(f, Loader=yaml.CLoader)


model = io.load_ply(model_mpath.format(objId))

numImages = len(gt_info)
for im_id in range(0,numImages):
    # print rgb_in_mpath.format(objId,im_id)
    maskData = np.zeros((h, w), dtype=np.uint8)

    # rgb = io.load_im(rgb_in_mpath.format(objId,im_id))
    [r,t]= getRotTrans(gt_info,im_id,objId)
    # print r
    # print t

    m_rgb = renderer.render(model, (w,h), cam_mat, r, t,shading='phong', mode='rgb',bg_color=(1.0, 1.0, 1.0, 0.0))

    img = Image.fromarray(m_rgb)
    # img = img.convert('1')
    # img.show();
    fileName = rgb_in_mpath.format(objId,im_id)
    img.save(fileName)
    img.close()
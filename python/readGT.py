import ruamel.yaml as yaml
import numpy as np
import inout as io
from PIL import Image
import renderer
import os
import time

from utils import *
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'test/{:02d}/rgb/{:04d}.png'
depth_in_mpath = base_path+ 'test/{:02d}/depth/{:04d}.png'

model_mpath = base_path + 'models/obj_{:02d}.ply' # Already transformed
seg_path = base_path+'test/{:02d}/seg/'



objId =4
scene_id =4;

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'


w, h = 640, 480


with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gt_info = yaml.load(f, Loader=yaml.CLoader)


model = io.load_ply(model_mpath.format(objId))

numImages = len(gt_info)
for im_id in range(0,numImages):
    print rgb_in_mpath.format(objId,im_id)
    maskData = np.zeros((h, w), dtype=np.uint8)

    rgb = io.load_im(rgb_in_mpath.format(objId,im_id))
    depth = io.load_im(depth_in_mpath.format(objId,im_id))

    [r,t]= getRotTrans(gt_info,im_id,objId)
    # r= np.ones([3,3]).reshape([3,3])
    # t= np.zeros([3 ,1]).reshape([3,1])
    # r=np.array([[0.9839512652064699, -0.03596483783742432, -0.1747753933993974],
    # [-0.1594908777229661, -0.6164855469907529, -0.7710435981672308],
    # [-0.08001604602668291, 0.7865444048436425, -0.6123277974969091]]).reshape([3,3])
    # t=np.array([53.5579, -84.0299, 1023.84]).reshape([3,1])
    # print r
    # print t

    m_rgb = renderer.render(model, (w,h), cam_mat, r, t,shading='phong', mode='rgb')
    time.sleep(1);
    r1, g1, b1 = 0, 0, 0  # Original value
    r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with
    # r2, g2, b2 = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]


    red, green, blue =m_rgb[:, :, 0], m_rgb[:, :, 1], m_rgb[:, :, 2]
    mask = (red != r1) & (green != g1) & (blue != b1)
    m_rgb[:, :, :3][mask] = r2,g2,b2

    img = Image.fromarray(m_rgb)
    img = img.convert('1')
    # img.show();
    fileName = seg_path.format(objId)+'%04d.png'%(im_id)
    img.save(fileName)
    img.close()
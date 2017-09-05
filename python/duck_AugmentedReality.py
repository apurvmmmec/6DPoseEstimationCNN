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
seg_path = base_path+'test/{:02d}/seg_02_05/'
rayban_duck= base_path+'test/{:02d}/glasses/'
rayban_mpath = base_path + 'models/rayban.ply' # Already transformed



objId =9
scene_id =9;



# bbox_cens_path = 'output/bbox_cens.yml'

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'


w, h = 640, 480





with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gt_info = yaml.load(f, Loader=yaml.CLoader)


model = io.load_ply(model_mpath.format(objId))

rotY15=np.array([ 0.9848  ,       0  ,  0.1736,
         0,    1.0000 ,        0,
   -0.1736 ,        0 ,   0.9848]).reshape([3,3]);

rotZ90=np.array([ 0  ,   1 ,    0,
    -1  ,   0  ,   0,
     0  ,   0  ,   1]).reshape([3,3]);

glass = io.load_ply(rayban_mpath.format(objId))
glass['pts']=glass['pts']/14
glass['pts'] = np.dot(np.dot(glass['pts'],rotZ90) , rotY15)


glass['pts'][:,0] =glass['pts'][:,0]/1.2
glass['pts'][:,0] =glass['pts'][:,0]+13.0

glass['pts'][:,1] =glass['pts'][:,1]*1.2

glass['pts'][:,2] =glass['pts'][:,2]+17






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

    m_rgb = renderer.render(glass, (w,h), cam_mat, r, t,shading='phong', mode='rgb+depth')
    time.sleep(1);
    r1, g1, b1 = 0, 0, 0  # Original value
    r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with
    r2, g2, b2 = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    glass_behind_duck = np.where(m_rgb[1]>depth)
    # else:
    red, green, blue =m_rgb[0][:, :, 0], m_rgb[0][:, :, 1], m_rgb[0][:, :, 2]
    numPt = len(glass_behind_duck[0])
    for i in range(0,numPt):
        red[glass_behind_duck[0][i], glass_behind_duck[1][i]] = rgb[glass_behind_duck[0][i], glass_behind_duck[1][i], 0]
        green[glass_behind_duck[0][i], glass_behind_duck[1][i]] = rgb[glass_behind_duck[0][i], glass_behind_duck[1][i], 1]
        blue[glass_behind_duck[0][i], glass_behind_duck[1][i]] = rgb[glass_behind_duck[0][i], glass_behind_duck[1][i], 2]

    mask = (red == r1) & (green == g1) & (blue == b1)
    m_rgb[0][:, :, :3][mask] = rgb[:, :, :3][mask]

    img = Image.fromarray(m_rgb[0])
    # img = img.convert('1')
    # img.show();
    fileName = rayban_duck.format(objId)+'%04d.jpg'%(im_id)
    print fileName
    img.save(fileName)
    img.close()
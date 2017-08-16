import ruamel.yaml as yaml
import numpy as np
import inout as io
from PIL import Image
import renderer
import matplotlib.pyplot as plt
import os
from utils import *

scene_id =1;
objId = 1
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'test/{:02d}/rgb/{:04d}.png'
model_mpath = base_path + 'models/obj_{:02d}.ply' # Already transformed
seg_path = base_path+'test/{:02d}/seg/'

# if (not(os.path.isdir(seg_path.format(objId)))):
#     print("Making Segmentation Directory")
#     os.mkdir(seg_path.format(objId))

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'

#
#
#
# w, h = 640, 480
#
#
# with open(scene_gt_mpath.format(scene_id), 'r') as f:
#     gt_info = yaml.load(f, Loader=yaml.CLoader)
# #
#
# model = io.load_ply(model_mpath.format(objId))
#
# for im_id in range(0,1236):
#     print rgb_in_mpath.format(objId,im_id)
#     maskData = np.zeros((h, w), dtype=np.uint8)
#
#     rgb = io.load_im(rgb_in_mpath.format(objId,im_id))
#     [r,t]= getRotTrans(gt_info,im_id,objId)
#     print r
#     print t
#
#     m_rgb = renderer.render(model, (w,h), cam_mat, r, t,shading='phong', mode='rgb',surf_color=(1,1,1))
#     # I,J= np.where(m_rgb[:,:,:] != [0,0,0] )
#
#     r1, g1, b1 = 0, 0, 0  # Original value
#     r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with
#
#     red, green, blue = m_rgb[:, :, 0], m_rgb[:, :, 1], m_rgb[:, :, 2]
#     mask = (red != r1) & (green != g1) & (blue != b1)
#     m_rgb[:, :, :3][mask] = [r2, g2, b2]
#
#     # plt.imshow(m_rgb)
#     # # plt.show()
#     img = Image.fromarray(m_rgb)
#     # img.show()
#     # im.save('fig1_modified.png')
#     # img = Image.fromarray(m_rgb)
#     img = img.convert('1')
#     img.show();
#     # img.save(seg_path.format(objId,im_id))
#     img.close()
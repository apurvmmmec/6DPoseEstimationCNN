import ruamel.yaml as yaml
import numpy as np
import inout as io
from PIL import Image
import renderer
import matplotlib.pyplot as plt

base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'test/{:02d}/rgb/{:04d}.png'
model_mpath = base_path + 'models/obj_{:02d}.ply' # Already transformed
seg_path = base_path+'test/{:02d}/seg/{:04d}.png'

# bbox_cens_path = 'output/bbox_cens.yml'

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'


def project_pts(pts, K, R, t):
    assert(pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T

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


w, h = 640, 480

scene_id =1; #
# objId = 5 #Can
objId = 1#Duck



with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gts = yaml.load(f, Loader=yaml.CLoader)


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
model = io.load_ply(model_mpath.format(objId))

for im_id in range(0,1236):
    print rgb_in_mpath.format(objId,im_id)
    maskData = np.zeros((h, w), dtype=np.uint8)

    rgb = io.load_im(rgb_in_mpath.format(objId,im_id))
    [r,t]= getRotTrans(im_id,objId)
    print r
    print t
    # r=r.dot(rotY180)
    # r=r.dot(rotZ180)



    # pts_im = project_pts(model['pts'], K, r, t)
    # pts_im = np.round(pts_im).astype(np.int)
    #
    # numPts = pts_im.shape[0]
    #
    # for i in range(0,numPts,1):
    #     if( (pts_im[i,1] >0) and (pts_im[i,1] <h) and (pts_im[i,0] >0) and (pts_im[i,0] <w)):
    #         maskData[pts_im[i,1],pts_im[i,0]] = 255
    #
    # img = Image.fromarray(maskData)
    # img.show();
    # img.save(seg_path.format(objId,im_id))
    # img.close()
    m_rgb = renderer.render(model, (w,h), K, r, t,shading='phong', mode='rgb',surf_color=(1,1,1))
    # I,J= np.where(m_rgb[:,:,:] != [0,0,0] )

    r1, g1, b1 = 0, 0, 0  # Original value
    r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with

    red, green, blue = m_rgb[:, :, 0], m_rgb[:, :, 1], m_rgb[:, :, 2]
    mask = (red != r1) & (green != g1) & (blue != b1)
    m_rgb[:, :, :3][mask] = [r2, g2, b2]

    # plt.imshow(m_rgb)
    # # plt.show()
    img = Image.fromarray(m_rgb)
    # img.show()
    # im.save('fig1_modified.png')
    # img = Image.fromarray(m_rgb)
    img = img.convert('1')
    # img.show();
    img.save(seg_path.format(objId,im_id))
    img.close()
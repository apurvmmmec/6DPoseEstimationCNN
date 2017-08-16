import ruamel.yaml as yaml
import numpy as np
import inout as io
from PIL import Image
from genColors import getColors
from genLabels import  getLabels
from numpy.linalg import inv
from getBB3D import getBB3D

base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'test/{:02d}/rgb/{:04d}.png'
depth_in_mpath = base_path+ 'test/{:02d}/depth/{:04d}.png'

labelled_color_mpath = base_path+ 'test/{:02d}/125bins/labelled_colored/{:04d}.png'
labelled_mpath = base_path+ 'test/{:02d}/125bins/labelled/{:04d}.png'


seg_in_mpath = base_path+ 'test/{:02d}/seg/{:04d}.png'

model_mpath = base_path + 'models/obj_{:02d}.ply'  # Already transformed

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'

bin_dim = 5
def project_pts(pts, K, R, t):
    assert (pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T


def getRotTrans(sceneId, objId):
    # for im_id, gts_im in gts.items():
    gts_im = gts[sceneId]
    for gt in gts_im:
        if 'obj_id' in gt.keys():
            if (gt['obj_id'] == objId):
                # print gt['obj_id']

                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                    r = gt['cam_R_m2c']
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
                    t = gt['cam_t_m2c']

    return r, t


h,w = 480, 640

objCal = np.zeros((h, w, 3), dtype=np.float32)
colorPalette = getColors(bin_dim=bin_dim);
labels = getLabels(bin_dim);
# print labels

scene_id = 1;  # benchviseblue
# objId = 5 #Can
objId = 1  # Cat

with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gts = yaml.load(f, Loader=yaml.CLoader)


# maxExtent = np.array([0.100792,0.181796,0.193734]).reshape([3, 1]); #Can
# maxExtent = np.array([0.0774076 ,0.0856969 ,0.104429]).reshape([3, 1]); #Duck
# maxExtent = np.array([0.11848210000,0.14113240000,0.25822600000]).reshape([3,1]) #Iron
maxExtent=np.array([0.0758686,0.0775993,0.0917691 ]).reshape([3,1])


bb3D = getBB3D(maxExtent)

# Find the min and max bounds of bounding box along all 3 dimensions
minX = min(bb3D[:, 0]);
maxX = max(bb3D[:, 0]);
minY = min(bb3D[:, 1]);
maxY = max(bb3D[:, 1]);
minZ = min(bb3D[:, 2]);
maxZ = max(bb3D[:, 2]);

#Find the dimension of bin along each dimension
unitX = (maxX - minX) / bin_dim;
unitY = (maxY - minY) / bin_dim;
unitZ = (maxZ - minZ) / bin_dim;

for im_id in range(72,1236):
    print im_id
    objLabel = np.zeros([h, w, 3],dtype=np.uint8)
    labelImg = np.zeros([h,w],dtype=np.uint8)

    seg = io.load_im(seg_in_mpath.format(objId,im_id))
    depth = io.load_im(depth_in_mpath.format(objId,im_id))

    [r, t] = getRotTrans(im_id, objId)
    t = t / 1000
    print r
    print t
    objC = np.array([])
    numPt = 0;
    for col in range(0, 640, 1):
        for row in range(0, 480, 1):
            if (seg[row, col] > 0):

                d = depth[row, col];
                if(d != 0):
                    numPt = numPt + 1;
                    xc = (col+1  - 325.2611) * (d / (1000.0 * 572.4114));
                    yc = (row+1  - 242.04899) * (d / (1000 * 573.57043));
                    zc = np.float64(d) / 1000;

                    camC = np.array([xc, yc, zc, 1]).reshape([4, 1]);
                    toc = np.hstack((r, t))
                    toc = np.vstack((toc, np.array([0, 0, 0, 1]).reshape([1, 4])))
                    xo = inv(toc).dot(camC);

                    obX = xo[0, 0];
                    obY = xo[1, 0];
                    obZ = xo[2, 0];
                    if ((obX > minX) and (obX < maxX) and (obY > minY) and (obY < maxY) and
                            (obZ > minZ) and (obZ < maxZ)):
                        lx = np.ceil((obX - minX) / unitX);
                        if (lx == 0):
                            lx = 1

                        ly = np.ceil((obY - minY) / unitY);
                        if (ly == 0):
                            ly = 1

                        lz = np.ceil((obZ - minZ) / unitZ);
                        if (lz == 0):
                            lz = 1

                        strLabel = '%d%d%d' % (lx, ly, lz);
                        # print strLabel
                        idx, status = np.where(labels == (int(strLabel)))
                        l = idx[0]

                        objLabel[row, col, 0] = int(float(colorPalette[l][0]) * 255)
                        objLabel[row, col, 1] = int(float(colorPalette[l][1]) * 255)
                        objLabel[row, col, 2] = int(float(colorPalette[l][2]) * 255)

                        labelImg[row, col] = (l + 1)

    imgColorCoord = Image.fromarray(objLabel,'RGB');

    # imgColorCoord.save(labelled_color_mpath.format(objId,im_id));

    imgLabel = Image.fromarray(labelImg);
    # imgLabel.show()
    #
    # imgLabel.save(labelled_mpath.format(objId,im_id));

    # imgColorCoord.show();

    # print model_mpath.format(objId)

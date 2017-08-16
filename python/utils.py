import numpy as np


def getRotTrans(gt_info,im_id,objId):

    # for im_id, gts_im in gts.items():
    gts_im = gt_info[im_id]
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


rotY180=np.array([-1   ,  0  ,   0,
     0 ,    1 ,    0,
     0 ,    0 ,   -1]).reshape([3,3]);

rotZ180=np.array([-1   ,  0  ,   0,
     0 ,    -1 ,    0,
     0 ,    0 ,   1]).reshape([3,3]);


cam_mat = np.zeros((3, 3));
cam_mat[0, 0] = 572.41140
cam_mat[0, 2] = 325.2611
cam_mat[1, 1] = 573.57043
cam_mat[1, 2] = 242.04899
cam_mat[2, 2] = 1
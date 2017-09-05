import ruamel.yaml as yaml
import numpy as np
import inout as io
import os


from utils import *
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

model_info_path = base_path +'models/models_info.yml'
info_file_path= base_path+'test/{:02d}/info/info_{:05d}.txt'
img_w = 640
img_h = 480


objId =2
scene_id=2 ;
scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'


with open(model_info_path,'r') as modelInfoFile:
    modelInfo = yaml.load(modelInfoFile,Loader=yaml.CLoader)




with open(scene_gt_mpath.format(scene_id), 'r') as f:
    gt_info = yaml.load(f, Loader=yaml.CLoader)



numImages = len(gt_info)

for im_id in range(0,numImages):
    print im_id
    infoFile = open(info_file_path.format(scene_id,im_id), 'w')
    [r,t]= getRotTrans(gt_info,im_id,objId)
    # print 'image size'
    # print img_w , img_h
    # print 'ApeACCV'
    #
    # print 'rotatation:'
    # print r[0,0], r[0,1], r[0,2]
    # print r[1, 0], r[1, 1], r[1, 2]
    # print r[2, 0], r[2, 1], r[2, 2]
    # print 'centre:'
    # print t[0, 0]/1000, t[1, 0]/1000, t[2, 0]/1000
    # print 'extent:'
    # print modelInfo[objId]['size_x'] / 1000, modelInfo[objId]['size_y'] / 1000, modelInfo[objId]['size_z'] / 1000

    infoFile.writelines( 'image size\n')
    infoFile.writelines('%d,%d\n'%( img_w, img_h))
    infoFile.writelines( 'CanACCV\n')

    infoFile.writelines( 'rotation:\n')
    infoFile.writelines( '%f %f %f\n'%(r[0, 0], r[0, 1], r[0, 2]))
    infoFile.writelines( '%f %f %f\n'%(-r[1, 0], -r[1, 1], -r[1, 2]))
    infoFile.writelines( '%f %f %f\n'%(-r[2, 0], -r[2, 1], -r[2, 2]))
    infoFile.writelines( 'center:\n')



    infoFile.writelines('%f %f %f\n'%( t[0, 0] / 1000, -t[1, 0] / 1000, -t[2, 0] / 1000))
    infoFile.writelines( 'extent:\n')
    infoFile.writelines( '%f %f %f\n'%(modelInfo[objId]['size_x'] / 1000, modelInfo[objId]['size_y'] / 1000, modelInfo[objId]['size_z'] / 1000))



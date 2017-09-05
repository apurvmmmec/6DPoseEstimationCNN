import numpy as np
import os

base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
trainScenes_file_path = base_path + 'test/{:02d}/trainingScenes.txt'
testScenes_file_path = base_path + 'test/{:02d}/testScenes.txt'

rgb_in_mpath = base_path + 'test/{:02d}/rgb/{:04d}.png'
depth_in_mpath = base_path + 'test/{:02d}/depth/{:04d}.png'
seg_in_path = base_path + 'test/{:02d}/seg/{:04d}.png'
labelled_mpath = base_path + 'test/{:02d}/125bins/labelled/{:04d}.png'

labelled_output_mpath = base_path+'test/{:02d}/testResults_125bins/labelled/{:04d}.png'
seg_in_path = base_path + 'test/{:02d}/seg/{:04d}.png'
info_in_path = base_path + 'test/{:02d}/info/info_{:05d}.txt'



numTotal=   1195
objId = 4


def genTrainTestFiles():
    trainScenesFile = open(trainScenes_file_path.format(objId), 'w')
    testScenesFile = open(testScenes_file_path.format(objId), 'w')

    perms = np.random.permutation(numTotal)
    print perms
    numTrain = int(np.floor(numTotal*0.8))
    idx = perms[0:numTrain]
    idx=np.sort(idx)
    for i in idx:
        trainScenesFile.write('%d\n'%(i+1))
    idx = perms[numTrain:]
    idx=np.sort(idx)

    for i in idx:
        testScenesFile.write('%d\n'%(i+1))


def separateTrainTestImages():
    # trainScenesFile = open(trainScenes_file_path.format(objId), 'r')
    testScenesFile = open(testScenes_file_path.format(objId), 'r')

    # for line in trainScenesFile.readlines():
    #     print line
    #     # print 'cp '+ rgb_in_mpath.format(objId,int(line))+ base_path+'test/%02d/training/rgb/'%(objId)
    #
    #     os.system('cp '+ rgb_in_mpath.format(objId,int(line))+' '+ base_path+'test/%02d/training/rgb/'%(objId))
    #     os.system('cp '+ depth_in_mpath.format(objId,int(line))+' '+ base_path+'test/%02d/training/depth/'%(objId))
    #
    #     os.system('cp '+ seg_in_path.format(objId,int(line))+' '+ base_path+'test/%02d/training/seg/'%(objId))
    #
    #     os.system('cp '+ labelled_mpath.format(objId,int(line))+' '+ base_path+'test/%02d/training/labelled/'%(objId))


    for line in testScenesFile.readlines():
        print line
        # print 'cp '+ rgb_in_mpath.format(objId,int(line))+ base_path+'test/%02d/training/rgb/'%(objId)

        # os.system('cp '+ rgb_in_mpath.format(objId,int(line))+' '+ base_path+'test/%02d/test/rgb_noseg/'%(objId))
        # os.system('cp '+ depth_in_mpath.format(objId,int(line))+' '+ base_path+'test/%02d/test/depth_noseg/'%(objId))
        # os.system('cp '+ seg_in_path.format(objId,int(line))+' '+ base_path+'test/%02d/training/seg/'%(objId))
        # os.system('cp '+ labelled_output_mpath.format(objId,int(line))+' '+ base_path+'test/%02d/test/objLabels/'%(objId))
        os.system('cp '+ info_in_path.format(objId,int(line))+' '+ base_path+'test/%02d/test/info/'%(objId))
        print(    'cp '+ info_in_path.format(objId,int(line))+' '+ base_path+'test/%02d/test/info/'%(objId))


separateTrainTestImages()
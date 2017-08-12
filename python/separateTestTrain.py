import numpy as np


base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
trainScenes_file_path = base_path + 'test/{:02d}/trainingScenes.txt'
testScenes_file_path = base_path + 'test/{:02d}/testScenes.txt'

numTotal=   1195
objId = 5


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

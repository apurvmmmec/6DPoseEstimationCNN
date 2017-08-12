from PIL import Image
import numpy as np
import cv2


def getTestImage():

    base_path = './'
    rgbPatchPath = base_path + 'labelledRGB_patch/001_0001_05399.png'
    depthPatchPath = base_path + 'labelledDepth_patch/001_0001_05399.png'

    # rgbPatchPath = base_path + 'labelledRGB_patch_val/001_0012_06923.png'
    # depthPatchPath = base_path + 'labelledDepth_patch_val/001_0012_06923.png'

    # rgbPatchPath = base_path+ 'rgb/0001.png'
    # depPath =  base_path+ 'depth/0017.png'
    rgbImg = cv2.imread(rgbPatchPath,-1)
    rgbImg = cv2.resize(rgbImg, (227, 227))


    depImg = cv2.imread(depthPatchPath,-1)
    depImg = cv2.resize(depImg, (227, 227))

    rgbImg = rgbImg[..., ::-1]

    # rgbPatch =Image.fromarray(rgbImg)
    # rgbPatch.show()
    #
    # depPatch = Image.fromarray(depImg)
    # depPatch.show()

    # w=40
    # h=40
    #
    # width = 2 * w;
    # height = 2 * h;
    (rows,cols,channels)= rgbImg.shape
    # print(a)

    ct = 0;
    # for r in range(150,250):
    #     for c in range(260,380):
    testImg = np.ndarray([1, 227, 227, 4])

    # for r in range(150,151):
    #     for c in range(260,261):
    #         patchCenR = r + h - 1;
    #         patchCenC = c + w - 1;
    #         rgbSubImg = rgbImg[r:r + height, c:c + width];
    #         depSubImg = depImg[r:r + height, c:c + width];
    #         rgbR = cv2.resize(rgbSubImg, (227, 227))
    #         rgbR = rgbR.astype(np.float32)
    #         depR= cv2.resize(depSubImg,(227,227))
    #         # rgbPatch =Image.fromarray(rgbSubImg)
    #         # rgbPatch.show()
    #         # print rgbR.shape
    #         # print depR.shape
    #         testImg[ct, :, :, 0:3] = rgbR
    #         testImg[ct, :, :, 3] = depR
    #         ct=ct+1

            # print ct
    # rgbImg = rgbImg.astype(np.float32)
    # depImg = depImg.astype(np.float32)



    testImg[ct, :, :, 0:3] = rgbImg

    testImg[ct, :, :, 3] = depImg
    #         ct=ct+1
    return testImg



# a= np.zeros([4,3])
# a[0,0]=0
# a[1,1]=2
# a[1,2]=22
# a[2,2] =54
# print a
# i= np.argmax(a,1)
# print i
# getTestImage()
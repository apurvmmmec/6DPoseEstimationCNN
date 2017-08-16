import cv2
import numpy as np
from binCents import getBinCent
import math

bin_dim = 5
num_bins = pow(bin_dim,3)
objId  =5
imId =1

w = 640
h = 480

base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

rgb_in_mpath = base_path+ 'test/{:02d}/rgb/{:04d}.png'
depth_in_mpath = base_path+ 'test/{:02d}/depth/{:04d}.png'
labelled_color_mpath = base_path+ 'test/{:02d}/{:d}bins/labelled_colored/{:04d}.png'
labelled_mpath = base_path+ 'test/{:02d}/{:d}bins/labelled/{:04d}.png'
seg_in_mpath = base_path+ 'test/{:02d}/seg/{:04d}.png'
model_mpath = base_path + 'models/obj_{:02d}.ply'  # Already transformed
scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'

minDist2D = 10
minDist3D = 0.01
minDepth  = 300
minArea   = 400
inlierThreshold2D =20

K = np.zeros((3, 3));
K[0, 0] = 572.41140
K[0, 2] = 325.2611
K[1, 1] = 573.57043
K[1, 2] = 242.04899
K[2, 2] = 1

distCoeffs = np.zeros((5, 1))


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


 # brief Returns the minimal distance of a query point to a line formed by two other points.
 #     @param pt1 Point 1 to form the line.
 #     @param pt2 Point 2 to form the line.
 #     @param pt3 Query point.
 #     @return Distance.
def pointLineDistance(pt1, pt2, pt3):
    v1 = (pt2 - pt1)
    v2 = (pt3 - pt1)

    return cv2.norm( np.cross(v1,v2) ) / cv2.norm(v1)


def sample2DPoint(segMask):
    numTotal = len(segMask)
    perms = np.random.permutation(numTotal);
    randIndex = perms[0]
    pt2D = np.array(segMask[randIndex], float)
    return pt2D

def getMinDist(pointSet, point):
    minDist = -1.0
    dim = pointSet.shape[1]
    for i in range(0,pointSet.shape[0]):
        srcPt = pointSet[i,:]
        # print "Calculating distance between "
        # print srcPt
        # print point
        if(minDist<0):
            minDist=cv2.norm(srcPt,point)
        else:
            minDist = np.min([minDist,cv2.norm(srcPt, point)],axis=0)

        # print "MinDist for %d point is %f"%(dim,minDist)

    return  minDist

def samplePoint(objId, pts3D,pts2D,pt2D, pt2DLabel):
    violation=False;

    if(getMinDist(pts2D,pt2D) < minDist2D):
        violation = violation or True

    bin_id = pt2DLabel
    pt3D = getBinCent(objId, num_bins,bin_id)
    # print "Centroid"
    # print pt3D
    if (getMinDist(pts3D, pt3D) < minDist3D):
        violation = violation or True

    pts2D = np.vstack((pts2D,pt2D))
    pts3D = np.vstack((pts3D,pt3D))

    return  not(violation), pts2D,pts3D


def ranssac2D():

    segImg = cv2.imread(seg_in_mpath.format(objId, imId), -1)
    labelImg = cv2.imread(labelled_mpath.format(objId,num_bins,imId),-1)

    maskPixels = []
    for r in range(0 ,h):
        for c in range(0,640):
            if (segImg[r, c] > 0):
                maskPixels.append([r,c])

    segMask = np.array(maskPixels)


    # Now Sample for 2D-3D correspondance


    for i in range (1,2):
        points2D = np.empty((0, 2), float)
        points3D = np.empty((0, 3), float)

        pt2D = sample2DPoint(segMask)
        l1 = labelImg[int(pt2D[0]),int(pt2D[1])]

        _, points2D, points3D = samplePoint(objId, pts3D=points3D, pts2D=points2D,
                                            pt2D=pt2D,
                                            pt2DLabel=l1)

        pt2D = sample2DPoint(segMask)
        l2 = labelImg[int(pt2D[0]), int(pt2D[1])]
        status, points2D, points3D = samplePoint(objId, pts3D=points3D, pts2D=points2D,
                                            pt2D=pt2D,
                                            pt2DLabel=l2)
        if (not (status)):
            print "Status for %d correspondance %d" % (2, status)
            continue

        pt2D = sample2DPoint(segMask)
        l4 = labelImg[int(pt2D[0]), int(pt2D[1])]
        status, points2D, points3D = samplePoint(objId, pts3D=points3D, pts2D=points2D,
                                            pt2D=pt2D,
                                            pt2DLabel=l4)
        if(not(status)):
            print "Status for %d correspondance %d" % (3, status)
            continue

        pt2D = sample2DPoint(segMask)
        l3=labelImg[int(pt2D[0]),int(pt2D[1])]
        status, points2D, points3D = samplePoint(objId, pts3D=points3D, pts2D=points2D,
                                            pt2D=pt2D,
                                            pt2DLabel= l3)
        if (not (status)):
            print "Status for %d correspondance %d" % (4, status)
            continue


        # print "RANSAC Iteration %d Successful %d %d %d %d" % (i, l1,l2,l3,l4)
        if((l1==l2)| (l1==l3)|(l1==l4)|(l2==l3)|(l2==l4)|(l3==l4)):
            print "Same labels"
            continue

        if (pointLineDistance(points3D[0], points3D[1], points3D[2]) < minDist3D):
            print" 3D constraint violated"
            continue;
        if (pointLineDistance(points3D[0], points3D[1], points3D[3]) < minDist3D):
            print" 3D constraint violated"
            continue;
        if (pointLineDistance(points3D[0], points3D[2], points3D[3]) < minDist3D):
            print" 3D constraint violated"
            continue;
        if (pointLineDistance(points3D[1], points3D[2], points3D[3]) < minDist3D):
            print" 3D constraint violated"
            continue;



        # print  points3D
        # print points2D



        x = cv2.solvePnP(points3D, points2D, K, distCoeffs)
        rot = eulerAnglesToRotationMatrix(x[1])
        trans = x[2]


        # std::vector < cv::Point2f > projections;
        # cv::projectPoints(points3D, trans.first, trans.second, camMat, cv::Mat(), projections);
        projections, jac = cv2.projectPoints(points3D, x[1], x[2], K, distCoeffs)
        # print points2D
        # print imgpts

        foundOutlier = False
        for j in range (0 ,len(points2D)):
            if (cv2.norm(points2D[j] - projections[j]) < inlierThreshold2D):
                continue;
            foundOutlier = True;
            break;

        if (foundOutlier):
            print "Found Outlier"
            continue;
        print rot
        print trans

ranssac2D()

# r=[ 0.98310202 -0.0472039  -0.17686699;
#     -0.167686   -0.61976397 -0.76666403;
#     -0.0734263   0.78336698 -0.61720699];
# t=[ 0.0554482 ;
#     -0.08189324;
#     1.02571399];
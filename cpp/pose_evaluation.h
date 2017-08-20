/*
Copyright (c) 2016, TU Dresden
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "types.h"
#include "Hypothesis.h"

/** Method to calculate pose error metrics */

/**
 * @brief Calculates pose error metric according to Hinterstoisser et al. for objects that are not rotation symmetric.
 * 
 * The pose error is the mean vertex distance when transforming a point cloud of the object 
 * using the ground truth transformation and the estimated transformation.
 * 
 * @param pointCloud Point cloud of the object.
 * @param estHyp Estimated pose.
 * @param gtHyp Ground truth pose.
 * @return double Pose error.
 */
double evaluatePoseAligned(const std::vector<cv::Point3d>& pointCloud, 
    Hypothesis& estHyp, Hypothesis& gtHyp)
{
    double sumDist = 0.0;
  
    for(unsigned i = 0; i < pointCloud.size(); i++)
    {
	cv::Point3d estPt = estHyp.transform(pointCloud[i]);
	cv::Point3d gtPt = gtHyp.transform(pointCloud[i]);
	
	sumDist += cv::norm(cv::Mat(estPt), cv::Mat(gtPt));
    }
    
    return sumDist / pointCloud.size();
}

/**
 * @brief Caluclates the pose error similar to the pose error metric according to Hinterstoisser et al. but measuring the vertex distance in image space (2D).
 * 
 * The pose error is the mean distance of vertices projected into the image 
 * when transforming a point cloud of the object 
 * using the ground truth transformation and the estimated transformation.
 * 
 * @param pointCloud Point cloud of the object.
 * @param estHyp Estimated pose.
 * @param gtHyp Ground truth pose.
 * @return double Pose error.
 */
double evaluatePose2D(
    const std::vector<cv::Point3d>& pointCloud, 
    const std::pair<cv::Mat, cv::Mat>& estTrans, 
    const std::pair<cv::Mat, cv::Mat>& gtTrans)
{
    std::vector<cv::Point3f> pc;
    for(unsigned i = 0; i < pointCloud.size(); i++)
	pc.push_back(cv::Point3f(pointCloud[i].x, pointCloud[i].y, pointCloud[i].z));
  
    std::vector<cv::Point2f> projectionsEst, projectionsGT;
    cv::Mat camMat = GlobalProperties::getInstance()->getCamMat();
    
    cv::projectPoints(pc, estTrans.first, estTrans.second, camMat, cv::Mat(), projectionsEst);
    cv::projectPoints(pc, gtTrans.first, gtTrans.second, camMat, cv::Mat(), projectionsGT);    
    
    double sumDist = 0.0;

    for(unsigned i = 0; i < pointCloud.size(); i++)
	sumDist += cv::norm(projectionsEst[i] - projectionsGT[i]);
    
    return sumDist / pointCloud.size();
}

/**
 * @brief Calculates pose error metric according to Hinterstoisser et al. for objects that are rotation symmetric.
 * 
 * The pose error is the mean vertex distance when transforming a point cloud of the object 
 * using the ground truth transformation and the estimated transformation. For each vertex 
 * the closest correspondence is used.
 * 
 * @param pointCloud Point cloud of the object.
 * @param estHyp Estimated pose.
 * @param gtHyp Ground truth pose.
 * @return double Pose error.
 */
double evaluatePoseUnaligned(const std::vector<cv::Point3d>& pointCloud, 
    Hypothesis& estHyp, Hypothesis& gtHyp)
{
    double sumDist = 0.0;
  
    std::vector<cv::Point3d> pcEst(pointCloud.size());
    std::vector<cv::Point3d> pcGT(pointCloud.size());
    
    #pragma omp parallel for
    for(unsigned i = 0; i < pointCloud.size(); i++)
    {
	pcEst[i] = estHyp.transform(pointCloud[i]);
	pcGT[i] = gtHyp.transform(pointCloud[i]);
    }

    #pragma omp parallel for    
    for(unsigned i = 0; i < pcEst.size(); i++)
    {
        // for each vertex search for the closest corresponding vertex
	double minDist = cv::norm(cv::Mat(pcEst[i]), cv::Mat(pcGT[0]));
	
	for(unsigned j = 0; j < pcGT.size(); j++)
	{
	    double currentDist = cv::norm(cv::Mat(pcEst[i]), cv::Mat(pcGT[j]));
	    minDist = std::min(minDist, currentDist);
	}
	
	#pragma omp critical
	{
	    sumDist += minDist;
	}
    }
    
    return sumDist / pointCloud.size();
}


 /**
  * @brief Calculates pose error metric according to Hinterstoisser et al. Supports rotation symmetric and non symmetric objects.
  * 
  * The pose error is the mean vertex distance when transforming a point cloud of the object 
  * using the ground truth transformation and the estimated transformation.
  * 
  * @param pointCloud Point cloud of the object.
  * @param estHyp Estimated pose.
  * @param gtHyp Ground truth pose. 
  * @param rotationObject Is the object rotation symmetric?
  * @return double Pose error.
  */
 double evaluatePose(const std::vector<cv::Point3d>& pointCloud, 
    Hypothesis& estHyp, Hypothesis& gtHyp, bool rotationObject)
{
    if(rotationObject)
	return evaluatePoseUnaligned(pointCloud, estHyp, gtHyp);
    else
	return evaluatePoseAligned(pointCloud, estHyp, gtHyp);
}
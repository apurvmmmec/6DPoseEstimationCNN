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
#include "dataset.h"
#include "Hypothesis.h"
#include "detection.h"
#include "pose_evaluation.h"
#include "ransac2D.h"
#include "ransac3D.h"

/** Classes and methods for evaluating the pose estimation and intermediate outputs. */

/**
 * @brief Accumulator for evaluation statistics (for one object) across several frames.
 */
struct ObjEval
{
    ObjEval()
    {
	imgCount = 0;
	occlusion = trainDist = 0;
	objProb = bgProb = inliers = 0;
	recognitions = detections = 0;
	poseHS = pose5cm5deg = pose10cm10deg = 0;
    }
  
    jp::id_t objID;
    std::string objName;
  
    unsigned imgCount; // number of frames processed so far
    float occlusion; // sum off percentage occluded in each frame
    float trainDist; // sum of angular distance of each test frame to its closest training frame
    
    float objProb; // sum of probability (predicted by the forest) for this object within its ground truth mask
    float bgProb; // sum of probability (predicted by the forest) for this object outside its ground truth mask
    float inliers; // sum of percentage of inliers in the object coordinate prediction for this object within the ground truth mask
    
    unsigned recognitions; // how often has the object been correctly identified to be in a test frame (position is ignored)
    unsigned detections; // how often was the object correctly localized following a IoU criterion
    
    unsigned poseHS; // how often was the pose correctly estimated according to the criterion of Hinterstoisser et al. (average vertex distance)
    unsigned pose5cm5deg; // how often was the pose error below 5 deg and 5 cm
    unsigned pose10cm10deg; // how often was the pose error below 10 deg and 10 cm
    
    std::vector<float> angles; // list of rotation errors so far (used to calculate the median error later)
    std::vector<float> dists; // list of translation errors so far (used to calculate the median error later)
};

/**
 * @brief Prints a table of average statistics for several objects to the console.
 * 
 * @param objEval List of statistics. One entry per object.
 * @return void
 */
void printEval(const std::map<jp::id_t, ObjEval>& objEval)
{
    std::cout << std::endl << "---------------------------------------------------------" << std::endl;
    std::cout << "Object\t\t#Img\tOcc\tDist\tp(o)\tp(bg)\tInliers\tRecogn.\tDetect.\tHS\t5cm5deg\t10cm10deg\tMedAng.\tMedDist." << std::endl;
    
    for(const auto& evalPair : objEval)
    {
	ObjEval evalItem = evalPair.second;
      
	std::printf("%.7s\t\t", evalItem.objName.c_str());
	std::printf("%4i\t", evalItem.imgCount);
	
	std::printf("%5.1f\%\t", evalItem.occlusion / evalItem.imgCount * 100);
	std::printf("%5.1f\t", evalItem.trainDist / evalItem.imgCount);
	
	std::printf("%5.1f\%\t", evalItem.objProb / evalItem.imgCount * 100);
	std::printf("%5.1f\%\t", evalItem.bgProb / evalItem.imgCount * 100);
	std::printf("%5.1f\%\t", evalItem.inliers / evalItem.imgCount * 100);
	
	std::printf("%5.1f\%\t", evalItem.recognitions / (float) evalItem.imgCount * 100);
	std::printf("%5.1f\%\t", evalItem.detections / (float) evalItem.imgCount * 100);
	
	std::printf("%5.1f\%\t", evalItem.poseHS / (float) evalItem.imgCount * 100);
	std::printf("%5.1f\%\t", evalItem.pose5cm5deg / (float) evalItem.imgCount * 100);
	std::printf("%5.1f\%\t", evalItem.pose10cm10deg / (float) evalItem.imgCount * 100);
	
	if(evalItem.recognitions > 0)
	{
	    // calculate median errors
	    std::sort(evalItem.angles.begin(), evalItem.angles.end());
	    std::sort(evalItem.dists.begin(), evalItem.dists.end());
	  
	    std::printf("\t%5.1f\t", evalItem.angles[evalItem.recognitions / 2]);
	    std::printf("%.1fcm\t", evalItem.dists[evalItem.recognitions / 2] / 10);
	}
	else
	{
	    std::cout << "\t-\t-";
	}
	std::cout << std::endl;
    }
    std::cout << "---------------------------------------------------------" << std::endl;  
}

/**
 * @brief Stores average statistics for several objects to a file.
 * 
 * @param objEval Accumulated statistics. One entry per object.
 * @param fTime Average time for evaluating the random forest (per input frame).
 * @param rTime Average time for RANSAC pose optimization (per input frame).
 * @return void
 */
void storeEval(const std::map<jp::id_t, ObjEval>& objEval, float fTime, float rTime)
{
    GlobalProperties* gp = GlobalProperties::getInstance();
  
    for(const auto& evalPair : objEval)
    {
	const ObjEval& evalItem = evalPair.second;
      
	std::string resultsFileName = "./obj" + intToString(evalItem.objID) + ".txt";
	std::ofstream resultsFile;
	resultsFile.open(resultsFileName, std::ios::app);
	
	resultsFile << gp->fP.sessionString << 					// 0
	    "\t" << gp->fP.config <<						// 1
	    "\t" << (int) evalItem.objID <<					// 2
	    "\t" << evalItem.imgCount <<					// 3
	    "\t" << evalItem.occlusion / evalItem.imgCount <<			// 4
	    "\t" << evalItem.trainDist / evalItem.imgCount <<			// 5
	    "\t" << "inliers" <<						// 6
	    "\t" << evalItem.inliers / evalItem.imgCount <<			// 7
	    "\t" << "probs" <<							// 8
	    "\t" << evalItem.objProb / evalItem.imgCount <<			// 9
	    "\t" << evalItem.bgProb / evalItem.imgCount <<			// 10
	    "\t" << "detection" <<						// 11
	    "\t" << evalItem.recognitions / (float) evalItem.imgCount <<	// 12
	    "\t" << evalItem.detections / (float) evalItem.imgCount <<		// 13
	    "\t" << "poses" <<							// 14
	    "\t" << evalItem.poseHS / (float) evalItem.imgCount <<		// 15
	    "\t" << evalItem.pose5cm5deg / (float) evalItem.imgCount <<		// 16
	    "\t" << evalItem.pose10cm10deg / (float) evalItem.imgCount <<	// 17
	    "\t" << "times" << 							// 18
	    "\t" << fTime <<							// 19
	    "\t" << rTime << std::endl;						// 20
	resultsFile.close();	
    }  
}

/**
 * @brief Calculates average probabilities for an object inside and outside the ground truth segmentation.
 * 
 * @param objProb Output parameter. Average object probability within the ground truth segmentation.
 * @param bgProb Output parameter. Average object probability outside the ground truth segmentation.
 * @param probabilities Probability map for the object.
 * @param seg Ground truth segmentation of the object.
 * @return void
 */
void evalProbabilities(
    float& objProb,
    float& bgProb,
    const jp::img_stat_t& probabilities,
    const jp::img_id_t& seg)
{
    objProb = 0;
    bgProb = 0;
    unsigned objPx = 0;
    unsigned bgPx = 0;
    
    for(unsigned x = 0; x < seg.cols; x++)
    for(unsigned y = 0; y < seg.rows; y++)
    {
	float curProb = probabilities(y, x);
      
	if(seg(y, x))
	{
	    objProb += curProb;
	    objPx++;
	}
	else
	{
	    bgProb += curProb;
	    bgPx++;
	}
    }
    
    objProb /= objPx;
    bgProb /= bgPx;  
}

/**
 * @brief Calculates the number of inliers in the object coordinate prediction of a forest. 
 * 
 * The area outside the ground truth segmentation (implicitly given by ground truth object coordinates) is ignored.
 * 
 * @param objGT Ground truth object coordinates.
 * @param forest The forest that made the prediction.
 * @param leafImgs Forest prediction. One leaf image per tree. For each pixel the index of the leaf the input patch ended up in.
 * @param objID ID of the object the prediction should be evaluated for.
 * @param inlierThreshold Tolerated distance in mm between predicted object coordinate and ground truth.
 * @return float Fraction of inliers within the ground truth segmentation.
 */
float evalObjectCoordinates(
    const jp::img_coord_t& objGT,
    const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
    const std::vector<jp::img_leaf_t>& leafImgs,
    jp::id_t objID,
    float inlierThreshold)
{
    unsigned inlierCount = 0;
    unsigned predCount = 0;  
  
    for(unsigned x = 0; x < objGT.cols; x++)
    for(unsigned y = 0; y < objGT.rows; y++)
    for(unsigned t = 0; t < leafImgs.size(); t++)
    {
	if(!jp::onObj(objGT(y, x))) continue; // inside/outside ground truth segmentation?
      
	size_t leafIndex = leafImgs[t](y, x);
	const std::vector<jp::mode_t>* modes = forest[t].getModes(leafIndex, objID);
      
	for(unsigned m = 0; m < modes->size(); m++)
	{
	    jp::coord3_t pred = modes->at(m).mean;
	    if(!jp::onObj(pred)) continue; // ignore if nothing has been predicted
	    
	    predCount++;
	    if(cv::norm(objGT(y, x) - pred) < inlierThreshold)
		inlierCount++;
	}
    }

    return inlierCount / (float) std::max(predCount, 1u);
}

/**
 * @brief Returns the minimal angular distances (in deg) between the given ground truth (test) pose and a list of other (training) poses.
 * 
 * @param info Input ground truth pose. Angular distances are calculated relative to this pose.
 * @param training List of poses where the minimal angular distance should be calculated.
 * @return double Minimal angular distance in deg.
 */
double distanceToClosestNeighbor(jp::info_t info, const std::vector<jp::info_t>& training)
{
    Hypothesis hyp(info.rotation, cv::Point3d(0, 0, 0));
    
    double bestResult = 0;
    bool first = true;
    
    for(int i = 0; i < training.size(); i++)
    {
	Hypothesis hypSet(training[i].rotation, cv::Point3d(0, 0, 0));
	
	double dist = hypSet.calcAngularDistance(hyp);
	
	if(first || bestResult > dist)
	{
	    first = false;
	    bestResult = dist;
	}
    }
    
    return bestResult;
}    

/**
 * @brief Holds the evaluation statistics of one pose estimate (for one frame).
 */
struct ObjStat
{
    ObjStat()
    {
	recognized = 0;
	
	inliers = objProb = bgProb = 0;
	
	transErrorX = transErrorY = transErrorZ = 0;
	rotError = 0;
	projError = 0;
	
	overlap = 0;
	poseError = 0;
	
	trainDist = occlusion = 0;
	
	hypInliers = hypSamples = 0;
	hypLikelihood = 0;
    }
    
  
    bool recognized; // was the object reported to be in the frame? position is ignored
    
    float inliers; // percentage of object coordinate inliers within the ground truth segmentation
    float objProb; // average probability (as predicted by the forest) for this object within the ground truth segmentation
    float bgProb; // average probability (as predicted by the forest) for this object outside the ground truth segmentation
    
    float transErrorX; // pose translational error in the X component in mm
    float transErrorY; // pose translational error in the Y component in mm
    float transErrorZ; // pose translational error in the Z component in mm
    float rotError; // pose rotational error in deg
    float projError; // average pixel distance of projected vertex distances (similar to criterion of Hinterstoisser et al. but measured in 2d image space)
    
    float overlap; // 2D bounding box overlap (intersection over union, IoU)
    float poseError; // average distance of transformed vertices (criterion of Hinterstoisser et al.)
    
    float trainDist; // angular distance of the test frame to the closest training image
    float occlusion; // fraction of the object area occluded
    
    int hypInliers; // number of inliers as estimated by RANSAC for this pose hypothesis
    int hypSamples; // number of pixels that where looked at by RANSAC for this pose hypothesis
    float hypLikelihood; // likelihood of this pose hypothesis (optimization using uncertainty)
};

/**
 * @brief Method that completely evaluates pose estimates for all objects and intermediate forest outputs for one frame.
 * 
 * All statistics are added to the objEval list of the corresponding object. 
 * The statistics are also writen to console and to a file (if program runs in no display mode (-nD))
 * 
 * @param imageNumber Number of the current frame.
 * @param objEval Output parameter. List of statistics accumulators (one per object).
 * @param estPoses List of estimated poses (one per object).
 * @param forest Random forest that made the predictions.
 * @param leafImgs Random forest prediction (one leaf image per tree, for each pixel the index of the leaf the input patch ended up in).
 * @param testInfos Ground truth poses of all objects in this frame.
 * @param trainingInfos List of training poses for all objects.
 * @param probabilities Probability maps for each object (as predicted by the forest).
 * @param gtObjImgs List of ground truth object coordinates. One per object.
 * @param gtSegImgs List of ground truth segmentations. One per object.
 * @param bb3Ds List of 3D bounding boxes. One per object.
 * @param pointClouds List of point clouds. One per object.
 * @param poseThresholdIoU Threshold for accepting localization based on 2d bounding box overlap.
 * @param poseThresholdHS Threshold for accepting pose estimation according to the criterion of Hinterstoisser et al.
 * @param poseThresholdTrans Threshold on translational error for accepting pose estimations.
 * @param poseThresholdRot Threshold on rotational error for accepting pose estimations.
 * @return void
 */
template<class T>
void evalObjectsComplete(
    unsigned imageNumber,
    std::map<jp::id_t, ObjEval>& objEval,
    std::map<jp::id_t, typename T::TransHyp>& estPoses,
    const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
    const std::vector<jp::img_leaf_t>& leafImgs,
    const std::vector<jp::info_t>& testInfos,
    const std::vector<std::vector<jp::info_t>>& trainingInfos,
    const std::vector<jp::img_stat_t>& probabilities,
    const std::vector<jp::img_coord_t>& gtObjImgs,
    const std::vector<jp::img_id_t>& gtSegImgs,
    const std::vector<std::vector<cv::Point3f>>& bb3Ds, 
    const std::vector<std::vector<cv::Point3d>>& pointClouds,
    float poseThresholdIoU,
    float poseThresholdHS, 
    float poseThresholdTrans,
    float poseThresholdRot)
{
    GlobalProperties* gp = GlobalProperties::getInstance();
  
    // print a table of statistics to the console
    std::cout << "Object\t\tOcc.\tDist.\tp(o)\tp(bg)\tInliers\tRecogn.\tDetect.\tHS\t5cm5deg\t10cm10deg\tAng.\tDist." << std::endl;
    unsigned recognitionCount = 0; // how often have objects been correctly recognized?
    
    for(unsigned o = 0; o < testInfos.size(); ++o)
    {
	if(!testInfos[o].visible)
	{
	    // skip if we look for a specific object but not this one, skip not if we look for all objects
	    if(gp->tP.searchObject > 0 && gp->tP.searchObject != o+1) 
		continue;
	    
	    // check recognition	  
	    if(!gp->tP.displayWhileTesting)
	    {  
		std::string errorsFileName = "./obj" + intToString(o+1) + "_details.txt";
		std::ofstream errorsFile;
		errorsFile.open(errorsFileName, std::ios::app);
		    
		std::string posesFileName = "./obj" + intToString(o+1) + "poses.txt";
		std::ofstream posesFile;
		posesFile.open(posesFileName, std::ios::app);

		if(estPoses.find(o+1) == estPoses.end())
		{
		    // not visible, not recognized
		    errorsFile << gp->fP.sessionString <<	// 0 - identifier of the forest 
			"\t" << imageNumber <<			// 1 - number of the current frame
			"\t" << false <<			// 2 - is the object in the frame?
			"\t" << 0 << 				// 3 - angular distance of this frame to the closest training image
			"\t" << 100 << 				// 4 - amount of object area occluded
			"\t" << 0 << 				// 5 - fraction of object coordinate inliers within the ground truth segmentation
			"\t" << 0 << 				// 6 - average probability for this object within the ground truth segmentation
			"\t" << 0 << 				// 7 - average probability for this object outside the ground truth segmentation
			"\t" << false << 			// 8 - has the object been recognized by RANSAC?
			"\t" << 0 <<				// 9 - 2D bounding box overlap
			"\t" << 0 << 				// 10 - X component of translational pose error (mm)
			"\t" << 0 <<				// 11 - Y component of translational pose error (mm)
			"\t" << 0 << 				// 12 - Z component of translational pose error (mm)
			"\t" << 0 << 				// 13 - rotational pose error (deg)
			"\t" << 0 <<				// 14 - average error of projected vertices (2D)
			"\t" << 0 <<				// 15 - average error of transformed vertives (3D)
			"\t" << 0 <<				// 16 - inliers identified by RANSAC
			"\t" << 0 <<				// 17 - pixel count looked at by RANSAC
			"\t" << 0 <<				// 18 - likelihood of the pose determined by RANSAC
			std::endl;		
			
		    posesFile << imageNumber <<		// 0 - image number
			"\t" << 0 <<			// 1 - pose rotation, rodrigues vector, 1st componend
			"\t" << 0 <<			// 2 - pose rotation, rodrigues vector, 2nd componend
			"\t" << 0 <<			// 3 - pose rotation, rodrigues vector, 3rd componend
			"\t" << 0 << 			// 4 - pose translation, X in mm
			"\t" << 0 << 			// 5 - pose translation, Y in mm
			"\t" << 0 << 			// 6 - pose translation, Z in mm
			std::endl;
		}
		else
		{
		    // not visible, but recognized (false positive)
		    errorsFile << gp->fP.sessionString <<	// 0 - description see above
			"\t" << imageNumber <<			// 1
			"\t" << false <<			// 2
			"\t" << 0 << 				// 3
			"\t" << 1 << 				// 4
			"\t" << 0 << 				// 5
			"\t" << 0 << 				// 6
			"\t" << 0 << 				// 7
			"\t" << true << 			// 8
			"\t" << 0 <<				// 9
			"\t" << 0 << 				// 10
			"\t" << 0 <<				// 11
			"\t" << 0 << 				// 12
			"\t" << 0 << 				// 13
			"\t" << 0 <<				// 14
			"\t" << 0 <<				// 15
			"\t" << estPoses[o+1].inliers <<	// 16
			"\t" << estPoses[o+1].effPixels <<	// 17
			"\t" << estPoses[o+1].likelihood <<	// 18
			std::endl;
						
		    jp::jp_trans_t jpPose = jp::cv2our(estPoses[o+1].pose);
		    Hypothesis hypPose(jpPose.first, jpPose.second);
		    std::vector<double> vecPose = hypPose.getRodVecAndTrans();    
		    
		    posesFile << imageNumber << 			// 0 - description see above
			"\t" << vecPose[0] <<				// 1
			"\t" << vecPose[1] <<				// 2
			"\t" << vecPose[2] <<				// 3
			"\t" << vecPose[3] << 				// 4
			"\t" << vecPose[4] << 				// 5
			"\t" << vecPose[5] << 				// 6
			std::endl;
		}
		
		errorsFile.close();	  
		posesFile.close();	
	    }
	    
	    continue;
	}	
	
	// initialize statistic accumulator objects
	if(objEval.find(o+1) == objEval.end())
	{
	    objEval[o+1] = ObjEval();
	    objEval[o+1].objID = o+1;
	    objEval[o+1].objName = testInfos[o].name;
	}
	
	float trainDist = distanceToClosestNeighbor(testInfos[o], trainingInfos[o]);
	
	// test frame properties
	objEval[o+1].imgCount++;
	objEval[o+1].occlusion += testInfos[o].occlusion;
	objEval[o+1].trainDist += trainDist;

	std::printf("%.7s\t\t", testInfos[o].name.c_str());
	std::printf("%4.1f\%\t", testInfos[o].occlusion * 100);
	std::printf("%4.1f\t", trainDist);

	// collect pose statistics
	ObjStat objStat;
	objStat.trainDist = trainDist;	
	objStat.occlusion = testInfos[o].occlusion;
	
	// probability evaluation
	float objProb, bgProb;
	evalProbabilities(objProb, bgProb, probabilities[o], gtSegImgs[o]);
	
	std::printf("%4.1f\%\t", objProb * 100);
	objEval[o+1].objProb += objProb;
	objStat.objProb = objProb;
	
	std::printf("%4.1f\%\t", bgProb * 100);
	objEval[o+1].bgProb += bgProb;
	objStat.bgProb = bgProb;
	
	// forest inlier evaluation
	float inlierRate = evalObjectCoordinates(gtObjImgs[o], forest, leafImgs, o+1, gp->tP.ransacInlierThreshold3D);
	
	std::printf("%4.1f\%\t", inlierRate * 100);
	objEval[o+1].inliers += inlierRate;
	objStat.inliers = inlierRate;
	
	// check recognition	    
	if(estPoses.find(o+1) == estPoses.end())
	{
	    // not recognized
	    std::cout << REDTEXT("no\tno\tno\tno\tno") << "\t\t-\t-";
	    objStat.recognized = false;
	}
	else
	{
	    // correctly recognized
	    std::cout << GREENTEXT("yes") << "\t";
	    recognitionCount++;
	    objStat.recognized = true;
	
	    objEval[o+1].recognitions++;
	    
	    // check detection
	    cv::Rect estBB2D = getBB2D(gp->fP.imageWidth, gp->fP.imageHeight, bb3Ds[o], gp->getCamMat(), estPoses[o+1].pose);
	    cv::Rect gtBB2D = getBB2D(gp->fP.imageWidth, gp->fP.imageHeight, bb3Ds[o], gp->getCamMat(), jp::our2cv(testInfos[o]));
	    objStat.overlap = getIoU(estBB2D, gtBB2D);
	    
	    if(objStat.overlap > poseThresholdIoU)
	    {
		std::cout << GREENTEXT("yes") << "\t";
		objEval[o+1].detections++;
	    }
	    else
		std::cout << REDTEXT("no") << "\t";
	    
	    //check hinterstoisser metric
	    jp::jp_trans_t pose = jp::cv2our(estPoses[o+1].pose);
		
	    Hypothesis hyp = Hypothesis(pose.first, pose.second);
	    Hypothesis gtH = Hypothesis(testInfos[o]);
		
	    double poseError = evaluatePose(pointClouds[o], hyp, gtH, gp->tP.rotationObject);
	    double bbDia = cv::norm(testInfos[o].extent) * 1000.0;
	    objStat.poseError = poseError / bbDia;
	    
	    if(objStat.poseError < poseThresholdHS)
	    {
		std::cout << GREENTEXT("yes") << "\t";
		objEval[o+1].poseHS++;
	    }
	    else
		std::cout << REDTEXT("no") << "\t";
	    
	    //check jamies metric (5cm 5deg)
	    cv::Point3f transError = hyp.getTranslation() - gtH.getTranslation();
	    float rotError = hyp.calcAngularDistance(gtH);
	    
	    objStat.transErrorX = transError.x;
	    objStat.transErrorY = transError.y;
	    objStat.transErrorZ = transError.z;
	    objStat.rotError = rotError;
	    
	    if((cv::norm(transError) < poseThresholdTrans) && (rotError < poseThresholdRot))
	    {
		std::cout << GREENTEXT("yes") << "\t";
		objEval[o+1].pose5cm5deg++;
	    }
	    else
		std::cout << REDTEXT("no") << "\t";
	    
	    // relaxed version of jamies metrix
	    if((cv::norm(transError) < 2*poseThresholdTrans) && (rotError < 2*poseThresholdRot))
	    {
		std::cout << GREENTEXT("yes") << "\t\t";
		objEval[o+1].pose10cm10deg++;
	    }
	    else
		std::cout << REDTEXT("no") << "\t\t";
	    
	    objStat.projError = evaluatePose2D(pointClouds[o], jp::our2cv(pose), jp::our2cv(testInfos[o]));
	    
	    std::printf("%4.1f\t", rotError);
	    std::printf("%.1fcm\t", cv::norm(transError) / 10);
	    
	    objEval[o+1].angles.push_back(rotError);
	    objEval[o+1].dists.push_back(cv::norm(transError));
	    
	    // RANSAC statistics
	    objStat.hypInliers = estPoses[o+1].inliers;
	    objStat.hypSamples = estPoses[o+1].effPixels;
	    objStat.hypLikelihood = estPoses[o+1].likelihood;
	}
	
	// write results to file
	if(!gp->tP.displayWhileTesting)
	{
	    std::string errorsFileName = "./obj" + intToString(o+1) + "_details.txt";
	    std::ofstream errorsFile;
	    errorsFile.open(errorsFileName, std::ios::app);
	    
	    std::string posesFileName = "./obj" + intToString(o+1) + "poses.txt";
	    std::ofstream posesFile;
	    posesFile.open(posesFileName, std::ios::app);	 	    
	    
	    errorsFile << gp->fP.sessionString <<	// 0 - description see above
		"\t" << imageNumber <<			// 1
		"\t" << true <<				// 2
		"\t" << objStat.trainDist << 		// 3
		"\t" << objStat.occlusion << 		// 4
		"\t" << objStat.inliers << 		// 5
		"\t" << objStat.objProb << 		// 6
		"\t" << objStat.bgProb << 		// 7
		"\t" << (int) objStat.recognized << 	// 8
		"\t" << objStat.overlap <<		// 9
		"\t" << objStat.transErrorX << 		// 10
		"\t" << objStat.transErrorY <<		// 11
		"\t" << objStat.transErrorZ << 		// 12
		"\t" << objStat.rotError << 		// 13
		"\t" << objStat.projError <<		// 14
		"\t" << objStat.poseError <<		// 15
		"\t" << objStat.hypInliers <<		// 16
		"\t" << objStat.hypSamples <<		// 17
		"\t" << objStat.hypLikelihood <<	// 18
		std::endl;
		
		if( objStat.recognized)
		{
		    jp::jp_trans_t jpPose = jp::cv2our(estPoses[o+1].pose);
		    Hypothesis hypPose(jpPose.first, jpPose.second);
		    std::vector<double> vecPose = hypPose.getRodVecAndTrans();    
		    
		    posesFile << imageNumber << 		// 0 - description see above
			"\t" << vecPose[0] <<			// 1
			"\t" << vecPose[1] <<			// 2
			"\t" << vecPose[2] <<			// 3
			"\t" << vecPose[3] << 			// 4
			"\t" << vecPose[4] << 			// 5
			"\t" << vecPose[5] << 			// 6
			std::endl;
		}
		else
		{
		    posesFile << imageNumber << 	// 0 - description see above
			"\t" << 0 <<			// 1
			"\t" << 0 <<			// 2
			"\t" << 0 <<			// 3
			"\t" << 0 << 			// 4
			"\t" << 0 << 			// 5
			"\t" << 0 << 			// 6
			std::endl;		  
		}
		
	    errorsFile.close();
	    posesFile.close();
	}
	
	std::cout << std::endl;
    }

    // print number of objects that have been recognized but are not in the current frame
    std::cout << std::endl << "False positives: " << estPoses.size() - recognitionCount << std::endl; //
}
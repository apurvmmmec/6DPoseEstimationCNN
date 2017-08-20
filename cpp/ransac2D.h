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
#include "util.h"
#include "sampler2D.h"
#include "detection.h"
#include "stop_watch.h"
#include "Hypothesis.h"
#include <exception>

#include <nlopt.hpp>
#include <omp.h>
#include<iostream>
#include_next "binCentroids.h"

/**
 * @brief RANSAC class of finding poses based on object coordinate predictions in the RGB case.
 */
class Ransac2D
{
public:
    Ransac2D()
    {
    };
  
    /**
     * @brief Struct that bundels data that is held per pose hypothesis during optimization.
     */    
    struct TransHyp
    {
	TransHyp() {}
	TransHyp(int objID, jp::cv_trans_t pose) : pose(pose), objID(objID), inliers(0), maxPixels(0), effPixels(0), refSteps(0), likelihood(0) {}
      
	int objID; // ID of the object this hypothesis belongs to
	jp::cv_trans_t pose; // the actual transformation
	
	cv::Rect bb; // 2D bounding box of the object under this pose hypothesis
	
	// 2D - 3D inlier correspondences
	std::vector<cv::Point3f> inliers3D; // object coordinate inliers
	std::vector<cv::Point2f> inliers2D; // pixel positions associated with the object coordinate inliers
	std::vector<const jp::mode_t*> inliersM; // object coordinate distribution modes associated with the object coordinate inliers
	
	int maxPixels; // how many pixels should be maximally drawn to score this hyp
	int effPixels; // how many pixels habe effectively drawn (bounded by projection size)
	
	int inliers; // how many of them were inliers
	float likelihood; // likelihood of this hypothesis (optimization using uncertainty)

	int refSteps; // how many iterations has this hyp been refined?
	
	/**
	 * @brief Returns a score for this hypothesis used to sort in preemptive RANSAC.
	 * 
	 * @return float Score.
	 */	
  	float getScore() const 	{ return inliers; }
  	
	/**
	 * @brief Fraction of inlier pixels as determined by RANSAC.
	 * 
	 * @return float Fraction of inliers.
	 */	
	float getInlierRate() const { return inliers / (float) effPixels; }
	
	/**
	 * @brief Operator used in sorting hypothesis. Compares scores.
	 * 
	 * @return bool True if this hypothesis' score is bigger.
	 */
	bool operator < (const TransHyp& hyp) const { return (getScore() > hyp.getScore()); } 
    };

    /**
     * @brief Data used in NLOpt callback loop.
     */
    struct DataForOpt
    {
	TransHyp* hyp; // pointer to the data attached to the hypothesis being optimized.
	Ransac2D* ransac; // pointer to the RANSAC object for access of various methods.
    };
    
    /**
     * @brief Thin out the inlier correspondences of the given hypothesis if there are too many. For runtime speed.
     * 
     * @param hyp Output parameter. Inlier correspondences stored in this hypothesis will the filtered.
     * @param maxInliers Maximal number of inlier correspondences to keep. Method does nothing if correspondences are fewer to begin with.
     * @return void
     */
    inline void filterInliers(
	TransHyp& hyp,
	int maxInliers)
    {
	if(hyp.inliers2D.size() < maxInliers) return; // maximum number not reached, do nothing
      		
	// filtered list of inlier correspondences
	std::vector<cv::Point3f> inliers3D;
	std::vector<cv::Point2f> inliers2D;
	std::vector<const jp::mode_t*> inliersM;
	
	// select random correspondences to keep
	for(unsigned i = 0; i < maxInliers; i++)
	{
	    int idx = irand(0, hyp.inliers2D.size());
	    
	    inliers2D.push_back(hyp.inliers2D[idx]);
	    inliers3D.push_back(hyp.inliers3D[idx]);
	    inliersM.push_back(hyp.inliersM[idx]);
	}
	
	hyp.inliers2D = inliers2D;
	hyp.inliers3D = inliers3D;
	hyp.inliersM = inliersM;
    }
    
    /**
     * @brief Calculates the likelihood of a 2D pixel location under a 3D multi-variate Gaussian. 
     * 
     * A volume will be cast from the pixel into camera space. The likelihood is the volume integral under the Gaussian.
     * The Gaussian must be given in camera coordinates.
     * 
     * @param x X component of the 2D pixel location.
     * @param y Y component of the 2D pixel location.
     * @param camMat 3x3 matrix of intrisic camera parameters.
     * @param mean Mean (3dim) of the Gaussian in camera coordinates.
     * @param covar 3x3 covariance matrix of the of the Gaussian in camera coordiantes.
     * @return double Likelihood of the volume cast by the pixel.
     */    
    inline double likelihood(int x, int y, const cv::Mat_<float>& camMat, const cv::Mat_<float>& mean, const cv::Mat_<float>& covar)
    {
	// calculate normalized image coordinates
	x = -(x - camMat.at<float>(0, 2));
	y = +(y - camMat.at<float>(1, 2));
	
	// calculate the pixel ray, we assume camera center 0 0 0 is the anchor for all points
	cv::Mat_<float> rayDir(3, 1);
	rayDir(2, 0) = -1000.f;
	rayDir(0, 0) = x * rayDir(2, 0) / camMat.at<float>(0, 0);
	rayDir(1, 0) = y * rayDir(2, 0) / camMat.at<float>(1, 1);
	
	rayDir /= cv::norm(rayDir);
	
	// center ray
	cv::Mat_<float> rayC(3, 1);
	rayC(2, 0) = 1.f;
	rayC(0, 0) = 0;
	rayC(1, 0) = 0;
	
	// rotate the pixel ray to the center ray (z-axis)
	cv::Mat_<float> axis = rayDir.cross(rayC); // rotation axis
	    
	float rotS = cv::norm(axis);
	float rotC = rayDir.dot(rayC);
	    
	cv::Mat_<float> crossM = cv::Mat_<float>::zeros(3, 3);
	crossM(1, 0) = axis(2, 0);
	crossM(2, 0) = -axis(1, 0);
	
	crossM(0, 1) = -axis(2, 0);
	crossM(2, 1) = axis(0, 0);		
	
	crossM(0, 2) = axis(1, 0);
	crossM(1, 2) = -axis(0, 0);		

	cv::Mat_<float> rotation = cv::Mat_<float>::eye(3, 3) + crossM + crossM * crossM * (1 - rotC) / rotS / rotS;		
	
	// apply ray rotation to the Gaussian
	cv::Mat_<float> rotMean = rotation * mean;
	cv::Mat_<float> rotCovar = rotation * covar * rotation.t();
		
	// easy access to inv covar matrix
 	cv::Mat_<float> invCov = rotCovar.inv();
	float a = invCov(0, 0);
	float b = invCov(0, 1);
	float c = invCov(0, 2);
	float d = invCov(1, 1);
	float e = invCov(1, 2);
	float f = invCov(2, 2);
	
	// constant factor of normal distribution
	double k = std::pow(2.0 * PI, -1.5) * std::pow(cv::determinant(rotCovar), -0.5);
	
	// easy access of mean
	double mx = rotMean(0, 0);
	double my = rotMean(1, 0);
	double mz = rotMean(2, 0);

	// gaussian parameters (x and y are zero)
	double gf = f/2.0;
	double gg = c*mx + e*my + f*mz;
	double gh = -(b*mx*my + c*mx*mz + e*my*mz + a*mx*mx/2 + d*my*my/2 + f*mz*mz/2);
	
	// return integral over z with factor z*z
	return k * std::sqrt(PI) * (2.0*gf + gg*gg) / (4.0*std::pow(gf, 2.5)) * std::exp(gg*gg / 4.0 / gf + gh);
    }    

    /**
     * @brief Recalculate the pose.
     * 
     * The hypothesis pose is recalculated using the associated inlier correspondences. The 2D bounding box of the hypothesis is also updated.
     * 
     * @param hyp Pose hypothesis to update.
     * @param camMat Camera matrix used to determine the new 2D bounding box.
     * @param imgWidth Width of the image (used for clamping the 2D bb).
     * @param imgHeight Height of the image (used for clamping the 2D bb).
     * @param bb3D 3D bounding box of the object associated with the hypothesis (used for determining the 2D bb).
     * @param maxPixels The maximal number of correspondences that should be used for recalculating the pose (for speed up).
     * @return void
     */    
    inline void updateHyp2D(TransHyp& hyp,
	const cv::Mat& camMat, 
	int imgWidth, 
	int imgHeight, 
	const std::vector<cv::Point3f>& bb3D,
	int maxPixels)
    {
	if(hyp.inliers2D.size() < 4) return;
	filterInliers(hyp, maxPixels); // limit the number of correspondences
      
	// recalculate pose
	cv::solvePnP(hyp.inliers3D, hyp.inliers2D, camMat, cv::Mat(), hyp.pose.first, hyp.pose.second, true, CV_EPNP);
	
	// update 2D bounding box
	hyp.bb = getBB2D(imgWidth, imgHeight, bb3D, camMat, hyp.pose);
    }
    
    
    cv::Point2f drawPointFromObjSegMask(const std::vector<cv::Point2f>& objPixels){
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator
        std::uniform_int_distribution<> distr(1, objPixels.size()); // define the range
        int rand_num =  distr(eng);
        return objPixels[rand_num-1];
    }
    
    std::vector<cv::Point2f> createObjPixelsList(const jp::img_id_t& segImg){
        std::vector<cv::Point2f> objPixels;
        for(unsigned x = 0; x < segImg.cols; x++)
            for(unsigned y = 0; y < segImg.rows; y++)
                if(segImg(y,x)  ==  1){
                    objPixels.push_back(cv::Point2f(x,y));
                }
        return objPixels;
    }
    
    /**
     * @brief Creates a list of pose hypothesis (potentially belonging to multiple objects) which still have to be processed (e.g. refined).
     * 
     * The method includes all remaining hypotheses of an object if there is still more than one, or if there is only one remaining but it still needs to be refined.
     * 
     * @param hypMap Map of object ID to a list of hypotheses for that object.
     * @param maxIt Each hypotheses should be at least this often refined.
     * @return std::vector< Ransac3D::TransHyp*, std::allocator< void > > List of hypotheses to be processed further.
     */    
    std::vector<TransHyp*> getWorkingQueue(std::map<int, std::vector<TransHyp>>& hypMap, int maxIt)
    {
	std::vector<TransHyp*> workingQueue;
      
	for(auto it = hypMap.begin(); it != hypMap.end(); it++)
	for(int h = 0; h < it->second.size(); h++)
	    if(it->second.size() > 1 || it->second[h].refSteps < maxIt) //exclude a hypothesis if it is the only one remaining for an object and it has been refined enough already
		workingQueue.push_back(&(it->second[h]));

	return workingQueue;
    }
    
    /**
     * @brief Main pose estimation function. Given a forest prediction it estimates poses of all objects.
     * 
     * Poses are stored in the poses member of this class.
     * 
     * @param probs Probability map for each object.
     * @param forest Random forest that did the prediction.
     * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
     * @param bb3Ds List of 3D object bounding boxes. One per object.
     * @return float Time the pose estimation took in ms.
     */    
    float estimatePose(
                       const std::vector<jp::img_id_t>& gtSegImgs,
                       const std::vector<jp::img_cordL_t>& gtLabelImgs,
	const std::vector<std::vector<cv::Point3f>>& bb3Ds)
    {
	GlobalProperties* gp = GlobalProperties::getInstance();
      
	int pnpMethod = CV_P3P; // pnp algorithm to be used to calculate initial poses from 4 correspondences
	float minDist2D = 10; // in px, initial pixel coordinates sampled to generate a hypothesis should be at least this far apart (for stability)
	float minDist3D = 10; // in mm, initial object coordinates sampled to generate a hypothesis should be at least this far apart (for stability)
	float minDepth = 300; // when estimating the seach radius for the initial pixel correspondences its assumed that the object cannot be nearer than this (in mm)
	float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)

	//set parameters, see documentation of GlobalProperties
	int maxIterations = gp->tP.ransacMaxDraws;
	float inlierThreshold2D = gp->tP.ransacInlierThreshold2D;
	float inlierThreshold3D = gp->tP.ransacInlierThreshold3D;
	int ransacIterations = gp->tP.ransacIterations;
	int refinementIterations = gp->tP.ransacRefinementIterations;
	int preemptiveBatch = gp->tP.ransacBatchSize;
	int maxPixels = gp->tP.ransacMaxInliers;
	int minPixels = gp->tP.ransacMinInliers;
	int refIt = gp->tP.ransacCoarseRefinementIterations;

	bool fullRefine = gp->tP.ransacRefine;
	
	int imageWidth = gp->fP.imageWidth;
	int imageHeight = gp->fP.imageHeight;
	
	cv::Mat camMat = gp->getCamMat();

	// create samplers for choosing pixel positions according to probability maps
        
        std::vector<cv::Point2f> objPixels = createObjPixelsList(gtSegImgs[0]);
        cv::Point2f ptDummy = drawPointFromObjSegMask(objPixels);
        std::vector<cv::Point3f> binCentroids = createCentVector();

        
		// hold for each object a list of pose hypothesis, these are optimized until only one remains per object
	std::map<int, std::vector<TransHyp>> hypMap;
	std::map<int, unsigned> drawMap; // holds for each object the number of hypothesis drawn including the ones discarded for constrain violation
	
	float ransacTime = 0;
	StopWatch stopWatch;
	
	// sample initial pose hypotheses

//	#pragma omp parallel for
	for(unsigned h = 0; h < ransacIterations; h++)
	{  
	    for(unsigned i = 0; i < maxIterations; i++)
	    {
		// 2D pixel - 3D object coordinate correspondences
		std::vector<cv::Point2f> points2D;
		std::vector<cv::Point3f> points3D;
	      
		cv::Rect bb2D(0, 0, imageWidth, imageHeight); // initialize 2D bounding box to be the full image
		
		// sample first point and choose object ID
		
//            int aa=(int)objID;
            
            cv::Point2f myPt = drawPointFromObjSegMask(objPixels);
            //
            int objID = 1;
		
		#pragma omp critical
//		{
//		    drawMap[objID]++;
//		}
		
		// sample first correspondence
		samplePoint(objID, points3D, points2D, myPt, minDist2D, minDist3D,gtLabelImgs, binCentroids);
		

            cv::Point2f myPt1 = drawPointFromObjSegMask(objPixels);

		// sample other points in search radius, discard hypothesis if minimum distance constrains are violated
		if(!samplePoint(objID, points3D, points2D, myPt1, minDist2D, minDist3D,gtLabelImgs, binCentroids))
		    continue;
            cv::Point2f myPt2 = drawPointFromObjSegMask(objPixels);

		if(!samplePoint(objID, points3D, points2D, myPt2, minDist2D, minDist3D,gtLabelImgs, binCentroids))
		    continue;
            cv::Point2f myPt3 = drawPointFromObjSegMask(objPixels);

		if(!samplePoint(objID, points3D, points2D, myPt3, minDist2D, minDist3D,gtLabelImgs, binCentroids))
		    continue;

		// check for degenerated configurations
		if(pointLineDistance(points3D[0], points3D[1], points3D[2]) < minDist3D) continue;
		if(pointLineDistance(points3D[0], points3D[1], points3D[3]) < minDist3D) continue;
		if(pointLineDistance(points3D[0], points3D[2], points3D[3]) < minDist3D) continue;
		if(pointLineDistance(points3D[1], points3D[2], points3D[3]) < minDist3D) continue;

		// reconstruct camera
		jp::cv_trans_t trans;
		cv::solvePnP(points3D, points2D, camMat, cv::Mat(), trans.first, trans.second, false, pnpMethod);
		
		std::vector<cv::Point2f> projections;
		cv::projectPoints(points3D, trans.first, trans.second, camMat, cv::Mat(), projections);
		
		// check reconstruction, 4 sampled points should be reconstructed perfectly
		bool foundOutlier = false;
		for(unsigned j = 0; j < points2D.size(); j++)
		{
		    if(cv::norm(points2D[j] - projections[j]) < inlierThreshold2D) continue;
		    foundOutlier = true;
		    break;
		}
		if(foundOutlier) continue;	    
		
		// create a hypothesis object to store meta data
		TransHyp hyp(objID, trans);
		
		// update 2D bounding box
		hyp.bb = getBB2D(imageWidth, imageHeight, bb3Ds[objID-1], camMat, hyp.pose);

		//check if bounding box collapses
		if(hyp.bb.area() < minArea)
		    continue;	    
		
		#pragma omp critical
		{
		    hypMap[objID].push_back(hyp);
		}

		break;
	    }
	}
	
	ransacTime += stopWatch.stop();
	std::cout << "Time after drawing hypothesis: " << ransacTime << "ms." << std::endl;

	// create a list of all objects where hypptheses have been found
	std::vector<int> objList;
	std::cout << std::endl;
	for(std::pair<int, std::vector<TransHyp>> hypPair : hypMap)
	{
	    std::cout << "Object " << (int) hypPair.first << ": " << hypPair.second.size() << " (drawn: " << drawMap[hypPair.first] << ")" << std::endl;
	    objList.push_back(hypPair.first);
	}
	std::cout << std::endl;

	// create a working queue of all hypotheses to process
	std::vector<TransHyp*> workingQueue = getWorkingQueue(hypMap, refIt);
	
	// main preemptive RANSAC loop, it will stop if there is max one hypothesis per object remaining which has been refined a minimal number of times
	while(!workingQueue.empty())
	{
	    // draw a batch of pixels and check for inliers, the number of pixels looked at is increased in each iteration
	    #pragma omp parallel for
	    for(int h = 0; h < workingQueue.size(); h++)
		countInliers2D(*(workingQueue[h]), binCentroids,gtLabelImgs,camMat, inlierThreshold2D, minArea, preemptiveBatch);
	    
	    // sort hypothesis according to inlier count and discard bad half
	    #pragma omp parallel for 
	    for(unsigned o = 0; o < objList.size(); o++)
	    {
		int objID = objList[o];
		if(hypMap[objID].size() > 1)
		{
		    std::sort(hypMap[objID].begin(), hypMap[objID].end());
		    hypMap[objID].erase(hypMap[objID].begin() + hypMap[objID].size() / 2, hypMap[objID].end());
		}
	    }
	    workingQueue = getWorkingQueue(hypMap, refIt);
	    
	    // refine
	    #pragma omp parallel for
	    for(int h = 0; h < workingQueue.size(); h++)
	    {
		updateHyp2D(*(workingQueue[h]), camMat, imageWidth, imageHeight, bb3Ds[workingQueue[h]->objID-1], maxPixels);
		workingQueue[h]->refSteps++;
	    }
	    
	    workingQueue = getWorkingQueue(hypMap, refIt);
	}

	ransacTime += stopWatch.stop();
	std::cout << "Time after preemptive RANSAC: " << ransacTime << "ms." << std::endl;

	poses.clear();	
	
	std::cout << std::endl << "---------------------------------------------------" << std::endl;
	for(auto it = hypMap.begin(); it != hypMap.end(); it++)
	for(int h = 0; h < it->second.size(); h++)
	{
	    std::cout << BLUETEXT("Estimated Hypothesis for Object " << (int) it->second[h].objID << ":") << std::endl;
	    
	    
	  
	    // store pose in class member
	    poses[it->second[h].objID] = it->second[h];
	    
	    std::cout << "Inliers: " << it->second[h].inliers;
	    std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
	    std::cout << "Refined " << it->second[h].refSteps << " times. " << std::endl;
	    std::cout << "---------------------------------------------------" << std::endl;
	}
	std::cout << std::endl;
	
	if(fullRefine)
	{
	    ransacTime += stopWatch.stop();
	    std::cout << "Time after final refine: " << ransacTime << "ms." << std::endl << std::endl;
	}	
	
	return ransacTime;
    }
    
private:
 
    /**
     * @brief Look at a certain number of pixels and check for inliers.
     * 
     * Inliers are determined by reprojecting the object coordinate predictions of the random forest and comparing to the original pixel positions.
     * 
     * @param hyp Hypothesis to check.
     * @param forest Random forest that made the object coordinate prediction
     * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
     * @param camMat 3x3 matrix of intrisic camera parameters.
     * @param inlierThreshold Allowed distance between object coordinate reprojection and original 2D position (in px).
     * @param minArea Abort if the 2D bounding box area of the hypothesis became too small (collapses).
     * @param pixelBatch Number of pixels that should be ADDITIONALLY looked at. Number of pixels increased in each iteration by this amount.
     * @return void
     */  
    inline void countInliers2D(
      TransHyp& hyp,
      std::vector<cv::Point3f> binCentroids,
      const std::vector<jp::img_cordL_t>& gtLabelImgs,
      const cv::Mat& camMat,
      float inlierThreshold,
      int minArea,
      int pixelBatch)
    {
	// reset data of last RANSAC iteration
	hyp.inliers2D.clear();
	hyp.inliers3D.clear();
	hyp.inliersM.clear();
	hyp.inliers = 0;

	// abort if 2D bounding box collapses
	if(hyp.bb.area() < minArea) return;
	
	// obj coordinate predictions are collected first and then reprojected as a batch
	std::vector<cv::Point3f> points3D;
	std::vector<cv::Point2f> points2D;
	std::vector<cv::Point2f> projections;
	std::vector<const jp::mode_t*> modeList;	

	hyp.effPixels = 0; // num of pixels drawn
	hyp.maxPixels += pixelBatch; // max num of pixels to be drawn	
	
	int maxPt = hyp.bb.area(); // num of pixels within bounding box
	float successRate = hyp.maxPixels / (float) maxPt; // probability to accept a pixel
	
	std::mt19937 generator;
	std::negative_binomial_distribution<int> distribution(1, successRate); // lets you skip a number of pixels until you encounter the next pixel to accept
	
	for(unsigned ptIdx = 0; ptIdx < maxPt;)
	{
	    hyp.effPixels++;
	    
	    // convert pixel index back to x,y position
	    cv::Point2f pt2D(
		hyp.bb.x + ptIdx % hyp.bb.width, 
		hyp.bb.y + ptIdx / hyp.bb.width);
	        
	    // each tree in the forest makes one or more predictions, collect all of them
//	    for(unsigned t = 0; t < forest.size(); t++)
//	    {
//		const std::vector<jp::mode_t>* modes = getModes(hyp.objID, pt2D, forest, leafImgs, t);
//		for(int m = 0; m < modes->size(); m++)
//		{
        cv::Point3f obj = getBinCentroid(binCentroids, gtLabelImgs,pt2D);
		    
		    // store 3D object coordinate - 2D pixel correspondence and associated distribution modes
		    points3D.push_back(obj);
		    points2D.push_back(pt2D);
//		    modeList.push_back(&(modes->at(m)));
		

	    // advance to the next accepted pixel
	    if(successRate < 1)
		ptIdx += std::max(1, distribution(generator));
	    else
		ptIdx++;
	}
	
	// reproject collected object coordinates
	if(points3D.empty()) return;
	cv::projectPoints(points3D, hyp.pose.first, hyp.pose.second, camMat, cv::Mat(), projections);
	
	// check for inliers
	for(unsigned p = 0; p < projections.size(); p++)
	{	    
	    if(cv::norm(points2D[p] - projections[p]) < inlierThreshold)
	    {
	        // keep inlier correspondences
		hyp.inliers2D.push_back(points2D[p]);
		hyp.inliers3D.push_back(points3D[p]);
//		hyp.inliersM.push_back(modeList[p]);
		hyp.inliers++; // keep track of the number of inliers (correspondences might be thinned out for speed later)
	    }
	}
    }  
  
    
    
    inline cv::Point3f getBinCentroid(
                                      const std::vector<cv::Point3f>& binCentroids,
                                      const std::vector<jp::img_cordL_t>& gtLabelImgs,
                                      const cv::Point2f& pt)
    {
        int label = (int)gtLabelImgs[0](pt.y,pt.x);
        cv::Point3f cent(0.0,0.0,0.0);
        if(label != 0){
            cent = binCentroids[label-1];
        }
        return cent;
    }

    /** 
     * @brief Return the minimal distance of a query point to a set of points.
     * 
     * @param pointSet Set of points.
     * @param point Query point.
     * @return double Distance.
     */    
    template<class T>
    inline double getMinDist(const std::vector<T>& pointSet, const T& point)
    {
	double minDist = -1.f;
      
	for(unsigned i = 0; i < pointSet.size(); i++)
	{
	    if(minDist < 0) 
		minDist = cv::norm(pointSet.at(i) - point);
	    else
		minDist = std::min(minDist, cv::norm(pointSet.at(i) - point));
	}
	
	return minDist;
    }
   
    /** 
     * @brief Return the maximal distance of a query point to a set of points.
     * 
     * @param pointSet Set of points.
     * @param point Query point.
     * @return double Distance.
     */     
    template<class T>
    inline double getMaxDist(const std::vector<T>& pointSet, const T& point)
    {
	double maxDist = -1.f;
      
	for(unsigned i = 0; i < pointSet.size(); i++)
	{
	    if(maxDist < 0) 
		maxDist = cv::norm(pointSet.at(i) - point);
	    else
		maxDist = std::max(maxDist, cv::norm(pointSet.at(i) - point));
	}
	
	return maxDist;
    }   
   
    /**
     * @brief Returns the minimal distance of a query point to a line formed by two other points.
     * 
     * @param pt1 Point 1 to form the line.
     * @param pt2 Point 2 to form the line.
     * @param pt3 Query point.
     * 
     * @return double Distance.
     */   
    inline double pointLineDistance(
	const cv::Point3f& pt1, 
	const cv::Point3f& pt2, 
	const cv::Point3f& pt3)
    {
	return cv::norm((pt2 - pt1).cross(pt3 - pt1)) / cv::norm(pt2 - pt1);
    }
    
    /**
     * @brief Sample a 2D pixel - 3D object coordinate correspondence at a given pixel.
     * 
     * The methods checks some constraints for the new correspondence and returns false if one is violated.
     * 1) The new pixel position should be sufficiently far from pixel positions sampled previously.
     * 2) The object coordiante should be sufficiently far from object coordinates sampled previously.
     * 
     * @param objID Object for which the correspondence should be sampled for.
     * @param pts3D Output parameter. List of object coordinates. A new one will be added by this method.
     * @param pts2D Output parameter. List of pixel positions. A new one will be added by this method.
     * @param pt2D Pixel position at which the correspondence should be sampled
     * @param forest Random forast that did the prediction.
     * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
     * @param minDist3D The new object coordinate should be at least this far from the previously sampled object coordinates (in mm).
     * @param minDist2D The new pixel position should be at least this far from the previously sampled pixel positions (in px).
     * @return bool Returns true of no contraints are violated by the new correspondence.
     */    
    inline bool samplePoint(
	int objID,
	std::vector<cv::Point3f>& pts3D, 
	std::vector<cv::Point2f>& pts2D, 
	const cv::Point2f& pt2D,
    float minDist2D,
	float minDist3D,
    const std::vector<jp::img_cordL_t>& gtLabelImgs,
    const std::vector<cv::Point3f>& binCentroids)
    {
	bool violation = false;
      
	if(getMinDist(pts2D, pt2D) < minDist2D) violation = violation || true; // check distance to previous pixel positions
      
    cv::Point3f pt3D = getBinCentroid(binCentroids, gtLabelImgs,pt2D);
	
	if(getMinDist(pts3D, pt3D) < minDist3D) violation = violation || true; // check distance to previous object coordinates
	
	pts2D.push_back(pt2D);
	pts3D.push_back(pt3D);
	
	return !violation;
    }

    
   
    public:
    std::map<int, TransHyp> poses; // Poses that have been estimated. At most one per object. Run estimatePose to fill this member.
};

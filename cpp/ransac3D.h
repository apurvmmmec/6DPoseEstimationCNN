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

#include <nlopt.hpp>
#include <omp.h>

#include "binCentroids.h"

using namespace std;
/**
 * @brief RANSAC class of finding poses based on object coordinate predictions in the RGB-D case.
 */
class Ransac3D
{
public:
    Ransac3D()
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
        
        std::vector<std::pair<cv::Point3d, cv::Point3d>> inlierPts; // list of object coordinate - camera coordinate correspondences that support this hypothesis
        std::vector<const jp::mode_t*> inlierModes; // for each correspondence above also the full distribution mode corresponding to the object coordiante
        
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
        Ransac3D* ransac; // pointer to the RANSAC object for access of various methods.
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
        if(hyp.inlierPts.size() < maxInliers) return; // maximum number not reached, do nothing
      		
        std::vector<std::pair<cv::Point3d, cv::Point3d>> inlierPts; // filtered list of inlier correspondences
        std::vector<const jp::mode_t*> inlierModes; // filtered list of associated object coordinate modes
        
        // select random correspondences to keep
        for(unsigned i = 0; i < maxInliers; i++)
        {
            int idx = irand(0, hyp.inlierPts.size());
            
            inlierPts.push_back(hyp.inlierPts[idx]);
            inlierModes.push_back(hyp.inlierModes[idx]);
        }
        
        hyp.inlierPts = inlierPts;
        hyp.inlierModes = inlierModes;
    }
    
    
    /**
     * @brief Calculates the likelihood of a camera coordinate under a multi-variate Gaussian. The Gaussian must be given in camera coordinates.
     *
     * @param eyePt Camera coordinate.
     * @param mean Mean (3dim) of the Gaussian in camera coordinates.
     * @param covar 3x3 covariance matrix of the of the Gaussian in camera coordiantes.
     * @return double
     */
    inline double likelihood(cv::Point3d eyePt, const cv::Mat_<float>& mean, const cv::Mat_<float>& covar)
    {
        // data conversion
        cv::Mat_<float> eyeMat(3, 1);
        eyeMat(0, 0) = eyePt.x;
        eyeMat(1, 0) = eyePt.y;
        eyeMat(2, 0) = eyePt.z;
        
        // evaluate Gaussian at the camera coordinate
        eyeMat = eyeMat - mean;
        eyeMat = eyeMat.t() * covar.inv() * eyeMat;
        
        float l = eyeMat(0, 0);
        l = std::pow(2 * PI, -1.5) * std::pow(cv::determinant(covar), -0.5) * std::exp(-0.5 * l);
        
        return l;
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
    inline void updateHyp3D(TransHyp& hyp,
                            const cv::Mat& camMat,
                            int imgWidth,
                            int imgHeight,
                            const std::vector<cv::Point3f>& bb3D,
                            int maxPixels)
    {
        if(hyp.inlierPts.size() < 4) return;
        //        filterInliers(hyp, maxPixels); // limit the number of correspondences
        
        // data conversion
        jp::jp_trans_t pose = jp::cv2our(hyp.pose);
        Hypothesis trans(pose.first, pose.second);
        
        // recalculate pose
        trans.refine(hyp.inlierPts);
        hyp.pose = jp::our2cv(jp::jp_trans_t(trans.getRotation(), trans.getTranslation()));
        
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
     * @param eyeData Camera coordinate image (point cloud) generated from the depth channel.
     * @param probs Probability map for each object.
     * @param forest Random forest that did the prediction.
     * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
     * @param bb3Ds List of 3D object bounding boxes. One per object.
     * @return float Time the pose estimation took in ms.
     */
    float estimatePose(
                       const jp::img_coord_t& eyeData,
                       const std::vector<std::vector<cv::Point3f>>& bb3Ds,
                       const std::vector<jp::img_id_t>& gtSegImgs,
                       const std::vector<jp::img_cordL_t>& gtLabelImgs)
    {
        GlobalProperties* gp = GlobalProperties::getInstance();
        
        //set parameters, see documentation of GlobalProperties
        int maxIterations = gp->tP.ransacMaxDraws;
        float minDist3D = 10; // in mm, initial coordinates (camera and object, respectively) sampled to generate a hypothesis should be at least this far apart (for stability)
        float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)
        
        float inlierThreshold3D = gp->tP.ransacInlierThreshold3D;
        int ransacIterations = gp->tP.ransacIterations;
        int refinementIterations = gp->tP.ransacRefinementIterations;
        int preemptiveBatch = gp->tP.ransacBatchSize;
        int maxPixels = gp->tP.ransacMaxInliers;
        int minPixels = gp->tP.ransacMinInliers;
        int refIt = gp->tP.ransacCoarseRefinementIterations;
        
        bool fullRefine = false;//gp->tP.ransacRefine;
        
        int imageWidth = gp->fP.imageWidth;
        int imageHeight = gp->fP.imageHeight;
        
        cv::Mat camMat = gp->getCamMat();
        
        // create samplers for choosing pixel positions according to probability maps
        
        std::vector<cv::Point2f> objPixels = createObjPixelsList(gtSegImgs[0]);
        cv::Point2f ptDummy = drawPointFromObjSegMask(objPixels);
        std::vector<cv::Point3f> binCentroids = createCentVector();
        
        // hold for each object a list of pose hypothesis, these are optimized until only one remains per object
        std::map<int, std::vector<TransHyp>> hypMap;
        
        float ransacTime = 0;
        StopWatch stopWatch;
        
        // sample initial pose hypotheses
#pragma omp parallel for
        for(unsigned h = 0; h < ransacIterations; h++)
            for(unsigned i = 0; i < maxIterations; i++)
            {
                // camera coordinate - object coordinate correspondences
                std::vector<cv::Point3f> eyePts;
                std::vector<cv::Point3f> objPts;
                
                cv::Rect bb2D(0, 0, imageWidth, imageHeight); // initialize 2D bounding box to be the full image
                
                // sample first point and choose object ID
                cv::Point2f myPt = drawPointFromObjSegMask(objPixels);
                //
                int objID = 1;
                                
                // sample first correspondence
                if(!samplePoint(objID, eyePts, objPts,myPt, eyeData, minDist3D,gtLabelImgs, binCentroids))
                    continue;
                
                // sample other points in search radius, discard hypothesis if minimum distance constrains are violated
                if(!samplePoint(objID, eyePts, objPts,myPt, eyeData, minDist3D,gtLabelImgs, binCentroids))
                    continue;
                
                if(!samplePoint(objID, eyePts, objPts,myPt, eyeData, minDist3D,gtLabelImgs, binCentroids))
                    continue;
                
                // reconstruct camera
                std::vector<std::pair<cv::Point3d, cv::Point3d>> pts3D;
                for(unsigned j = 0; j < eyePts.size(); j++)
                {
                    pts3D.push_back(std::pair<cv::Point3d, cv::Point3d>(
                                                                        cv::Point3d(objPts[j].x, objPts[j].y, objPts[j].z),
                                                                        cv::Point3d(eyePts[j].x, eyePts[j].y, eyePts[j].z)
                                                                        ));
                }
                
                Hypothesis trans(pts3D);
                
                // check reconstruction, sampled points should be reconstructed perfectly
                bool foundOutlier = false;
                for(unsigned j = 0; j < pts3D.size(); j++)
                {
                    if(cv::norm(pts3D[j].second - trans.transform(pts3D[j].first)) < inlierThreshold3D) continue;
                    foundOutlier = true;
                    break;
                }
                if(foundOutlier) continue;
                
                // pose conversion
                jp::jp_trans_t pose;
                pose.first = trans.getRotation();
                pose.second = trans.getTranslation();
                
                // create a hypothesis object to store meta data
                TransHyp hyp(objID, jp::our2cv(pose));
                
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
        
        ransacTime += stopWatch.stop();
        std::cout << "Time after drawing hypothesis: " << ransacTime << "ms." << std::endl;
        
        // create a list of all objects where hypptheses have been found
        std::vector<int> objList;
        std::cout << std::endl;
        for(std::pair<int, std::vector<TransHyp>> hypPair : hypMap)
        {
            std::cout << "Object " << (int) hypPair.first << ": " << hypPair.second.size() << std::endl;
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
                countInliers3D(*(workingQueue[h]),binCentroids,gtLabelImgs, eyeData, inlierThreshold3D, minArea, preemptiveBatch);
            
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
                updateHyp3D(*(workingQueue[h]), camMat, imageWidth, imageHeight, bb3Ds[workingQueue[h]->objID-1], maxPixels);
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
     * Inliers are determined by comparing the object coordinate prediction of the random forest with the camera coordinates.
     *
     * @param hyp Hypothesis to check.
     * @param forest Random forest that made the object coordinate prediction
     * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
     * @param eyeData Camera coordinates of the input frame (point cloud generated from the depth channel).
     * @param inlierThreshold Allowed distance between object coordinate predictions and camera coordinates (in mm).
     * @param minArea Abort if the 2D bounding box area of the hypothesis became too small (collapses).
     * @param pixelBatch Number of pixels that should be ADDITIONALLY looked at. Number of pixels increased in each iteration by this amount.
     * @return void
     */
    inline void countInliers3D(
                               TransHyp& hyp,
                               std::vector<cv::Point3f> binCentroids,
                               const std::vector<jp::img_cordL_t>& gtLabelImgs,
                               const jp::img_coord_t& eyeData,
                               float inlierThreshold,
                               int minArea,
                               int pixelBatch)
    {
        // reset data of last RANSAC iteration
        hyp.inlierPts.clear();
        hyp.inlierModes.clear();
        hyp.inliers = 0;
        
        // abort if 2D bounding box collapses
        if(hyp.bb.area() < minArea) return;
        
        // data conversion
        jp::jp_trans_t pose = jp::cv2our(hyp.pose);
        Hypothesis trans(pose.first, pose.second);
        
        hyp.effPixels = 0; // num of pixels drawn
        hyp.maxPixels += pixelBatch; // max num of pixels to be drawn
        
        int maxPt = hyp.bb.area(); // num of pixels within bounding box
        float successRate = hyp.maxPixels / (float) maxPt; // probability to accept a pixel
        
        std::mt19937 generator;
        std::negative_binomial_distribution<int> distribution(1, successRate); // lets you skip a number of pixels until you encounter the next pixel to accept
        
        for(unsigned ptIdx = 0; ptIdx < maxPt;)
        {
            // convert pixel index back to x,y position
            cv::Point2f pt2D(
                             hyp.bb.x + ptIdx % hyp.bb.width,
                             hyp.bb.y + ptIdx / hyp.bb.width);
            
            // skip depth holes
            if(eyeData(pt2D.y, pt2D.x)[2] == 0)
            {
                ptIdx++;
                continue;
            }
            
            // read out camera coordinate
            cv::Point3d eye(eyeData(pt2D.y, pt2D.x)[0], eyeData(pt2D.y, pt2D.x)[1], eyeData(pt2D.y, pt2D.x)[2]);
            
            hyp.effPixels++;
            
            // each tree in the forest makes one or more predictions, check all of them
            
            cv::Point3f obj = getBinCentroid(binCentroids, gtLabelImgs,pt2D);
            
            
            // inlier check
            if(cv::norm(eye - trans.transform(obj)) < inlierThreshold)
            {
                hyp.inlierPts.push_back(std::pair<cv::Point3d, cv::Point3d>(obj, eye)); // store
                hyp.inliers++; // keep track of the number of inliers (correspondences might be thinned out for speed later)
            }
            // advance to the next accepted pixel
            if(successRate < 1)
                ptIdx += std::max(1, distribution(generator));
            else
                ptIdx++;
        }
    }
    
    
    /**
     * @brief  Returns the centroid of the bin in which objecte coordinates of 3D projection of 2D point lies.
     *
     * @param objID Object for which to look up the object coordinate.
     * @param pt Pixel position to look up.
     * @return cv::Point3f Center of the mode with largest support.
     */
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
     * @brief Sample a camera coordinate - object coordinate correspondence at a given pixel.
     *
     * The methods checks some constraints for the new correspondence and returns false if one is violated.
     * 1) There should be no depth hole at the pixel
     * 2) The camera coordinate should be sufficiently far from camera coordinates sampled previously.
     * 3) The object coordinate prediction should not be empty.
     * 4) The object coordiante should be sufficiently far from object coordinates sampled previously.
     *
     * @param objID Object for which the correspondence should be sampled for.
     * @param eyePts Output parameter. List of camera coordinates. A new one will be added by this method.
     * @param objPts Output parameter. List of object coordinates. A new one will be added by this method.
     * @param pt2D Pixel position at which the correspondence should be sampled
     * @param forest Random forast that did the prediction.
     * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
     * @param eyeData Camera coordinates of the input frame (point cloud generated from the depth channel).
     * @param minDist3D The new camera coordinate should be at least this far from the previously sampled camera coordinates (in mm). Same goes for object coordinates.
     * @return bool Returns true of no contraints are violated by the new correspondence.
     */
    inline bool samplePoint(
                            int objID,
                            std::vector<cv::Point3f>& eyePts,
                            std::vector<cv::Point3f>& objPts,
                            cv::Point2f& myPt,
                            const jp::img_coord_t& eyeData,
                            float minDist3D,
                            const std::vector<jp::img_cordL_t>& gtLabelImgs,
                            const std::vector<cv::Point3f>& binCentroids)
    {
        cv::Point3f eye(eyeData(myPt.y, myPt.x)[0], eyeData(myPt.y, myPt.x)[1], eyeData(myPt.y, myPt.x)[2]); // read out camera coordinate
        if(eye.z == 0) return false; // check for depth hole
        double minDist = getMinDist(eyePts, eye); // check for distance to previous camera coordinates
        if(minDist > 0 && minDist < minDist3D) return false;
        
        //        cv::Point2f myPt = drawPointFromObjSegMask(objPixels)
        cv::Point3f obj = getBinCentroid(binCentroids, gtLabelImgs,myPt);
        
        
        //        cv::Point3f obj = getMode(objID, pt2D, forest, leafImgs); // read out object coordinate
        if(obj.x == 0 && obj.y == 0 && obj.z == 0) return false; // check for empty prediction
        minDist = getMinDist(objPts, obj); // check for distance to previous object coordinates
        if(minDist > 0 && minDist < minDist3D) return false;
        
        eyePts.push_back(eye);
        objPts.push_back(obj);
        
        return true;
    }
    
    
    
    
public:
    std::map<int, TransHyp> poses; // Poses that have been estimated. At most one per object. Run estimatePose to fill this member.
};

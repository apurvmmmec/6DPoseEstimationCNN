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

#include "properties.h"
#include "util.h"
#include "util_display.h"
#include "dataset.h"
#include "write_data.h"

#include "regression_tree.h"
#include "regression_classifier.h"

#include "regression_evaluation.h"
#include "pose_evaluation.h"

#include "pointCloudTools.h"
#include "detection.h"
#include "auto_context.h"
#include "stop_watch.h"

#include "ransac2D.h"
#include "ransac3D.h"

#include "binCentroids.h"

#include <fstream>

/** Main program for pose estimation using auto-context random forests. */

int main(int argc, const char* argv[])
{
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig(); // read the default config file and set parameters accordingly
    gp->parseCmdLine(argc, argv); // parse parameters given via command line
    
    std::string dataDir = "./";
    std::string setDir = dataDir + "test/"; // test data directory
    std::string trainingDir = dataDir + "training/"; // training data directory (used to find the closes training image to a test image for evaluation)
    
    // get data sub folders
    std::vector<std::string> testSets = getSubPaths(setDir);
    std::vector<std::string> trainingSets = getSubPaths(trainingDir);
    
    gp->fP.objectCount = trainingSets.size();
    
    // load some more meta data used in the evaluation process
    std::cout << "Loading Point Clouds and Infos" << std::endl;
    std::vector<cv::Scalar> colors = getColors();
    std::vector<std::vector<cv::Point3d>> pointClouds;
    std::vector<jp::info_t> infos;
    std::vector<std::vector<cv::Point3f>> bb3Ds;
    
    for(int i = 0; i < gp->fP.objectCount; i++)
    {
        // one point cloud per object
        std::vector<cv::Point3d> pointCloud;
        loadPointCloudGeneric("pc" + intToString(i + 1) + ".xyz", pointCloud, 1000);
        pointClouds.push_back(pointCloud);
        
        // one default info per object to get the object extent and a 3D bounding box
        jp::info_t info;
        jp::readData(dataDir + "info" + intToString(i+1) + ".txt", info);
        infos.push_back(info);
        
        bb3Ds.push_back(getBB3D(info.extent));
    }
    
    std::cout << std::endl << "Test Sequence: " << YELLOWTEXT(testSets[gp->tP.testObject-1]) << std::endl;
    std::cout << "Search Object: " << YELLOWTEXT(((gp->tP.searchObject > 0) ? intToString(gp->tP.searchObject) : "all")) << std::endl;
    
    std::cout << std::endl << BLUETEXT("Loading training sets ...") << std::endl;
    std::vector<jp::Dataset> trainDatasets;
    for(int trainIdx = 0; trainIdx < trainingSets.size(); trainIdx++)
        trainDatasets.push_back(jp::Dataset(trainingSets[trainIdx], trainIdx+1));
    
    std::cout << BLUETEXT("Loading test sets ...") << std::endl;
    std::vector<jp::Dataset> testDatasets;
    for(int testIdx = 0; testIdx < testSets.size(); testIdx++)
        testDatasets.push_back(jp::Dataset(testSets[testIdx], testIdx+1));
    
    //load training infos (used in evaluation)
    std::vector<std::vector<jp::info_t>> trainingInfos;
    for(unsigned o = 0; o < trainDatasets.size(); o++)
    {
        std::vector<jp::info_t> curInfos;
        for(unsigned i = 0; i < trainDatasets[o].size(); i++)
        {
            jp::info_t curInfo;
            trainDatasets[o].getInfo(i, curInfo);
            curInfos.push_back(curInfo);
        }
        trainingInfos.push_back(curInfos);
    }
    
    float poseThresholdHS = 0.1; // threshold * object diameter is the tolerance for the mean pose estimation error, measure introduced bei Hinterstoisser et al. in ACCV12 paper
    float poseThresholdTrans = 50; // threshold for translation error in mm
    float poseThresholdRot = 5; // threshold for rotation error in degree
    float poseThresholdIoU = 0.5; // threshold for 2D bounding box overlap error (intersection over union)
    
    StopWatch stopWatch;
    float avgForestTime = 0.f; // average time for the forest evaluation
    float avgRansacTime = 0.f; // average time for the ransac pose optimization
    unsigned imageCount = 0; // number of images that have been processed
    
    
    std::map<jp::id_t, ObjEval> objEval; // stores evaluation statistics per objects (accumulated over test images)
    
    // main testing loop
    for(unsigned i = 0; i < testDatasets[gp->tP.testObject-1].size(); i+=gp->tP.imageSubSample)
    {
        std::cout << std::endl << BLUETEXT(i << "th image:") << std::endl;
        std::cout << testDatasets[gp->tP.testObject-1].getFileName(i) << std::endl;
        
        // load ground truth poses including visibility info
        for(unsigned o = 0; o < 1; o++)
        {
            if(gp->tP.searchObject == -1 || gp->tP.searchObject == o + 1) // check whether the current object is searched for
                testDatasets[o].getInfo(i, infos[o]);
            else
                infos[o] = jp::info_t(false); // load no ground truth info if this object is not searched for
        }
        
        // load test image
        jp::img_bgr_t colorData;
        testDatasets[gp->tP.testObject-1].getBGR(i, colorData, true);
        
        jp::img_data_t inputData;
        inputData.seg = jp::img_id_t::ones(colorData.rows, colorData.cols); // test segmentation is the complete image
        inputData.labelData = std::vector<jp::img_label_t>(gp->fP.objectCount); // initialize auto-context feature channels
        inputData.coordData = std::vector<jp::img_coord_t>(gp->fP.objectCount);
        inputData.colorData = colorData;
        
        // load depth only in case it should be used (NULL for RGB setting)
        jp::img_depth_t depthData;
        jp::img_depth_t* depth = NULL;
        if(gp->fP.useDepth)
        {
            testDatasets[gp->tP.testObject-1].getDepth(i, depthData, true);
            depth = &depthData;
        }
        
        // do regression task
        stopWatch.init();
        
        // load ground truth data
        std::vector<jp::img_coord_t> gtObjImgs; // ground truth object coordinates
        std::vector<jp::img_id_t> gtSegImgs; // ground truth segmentation masks
        std::vector<jp::img_cordL_t> gtLabelImgs; // ground truth segmentation masks
        
        
        
        for(unsigned o = 0; o < infos.size(); o++)
        {
            if(!infos[o].visible) // load no ground truth for objects not visible
            {
                gtSegImgs.push_back(jp::img_id_t());
                gtObjImgs.push_back(jp::img_coord_t());
                gtLabelImgs.push_back(jp::img_cordL_t());
                
                continue;
            }
            
            jp::img_id_t gtSegImg;
            testDatasets[o].getSegmentation(i, gtSegImg);
            gtSegImgs.push_back(gtSegImg);
            
            jp::img_coord_t gtObjImg;
            testDatasets[o].getObj(i, gtObjImg);
            gtObjImgs.push_back(gtObjImg);
            
            jp::img_cordL_t gtLabelImg;
            testDatasets[o].getObjCoordLabelImage(i, gtLabelImg);
            gtLabelImgs.push_back(gtLabelImg);
        }
        
        // do pose estimation
        Ransac3D ransac3D; // RANSAC variant for RGB-D images
        	Ransac2D ransac2D; // RANSAC variant for RGB images
        float ransacTime;
        
        if(gp->fP.useDepth) //RGB-D case
        {
            // extract camera coordinate image (point cloud) from depth channel
            jp::img_coord_t eyeData;
            testDatasets[gp->tP.testObject-1].getEye(i, eyeData);
            
            // perform pose estimation based on forest output, poses are stored in RANSAC class
            ransacTime = ransac3D.estimatePose(
                                               eyeData,
                                               bb3Ds,gtSegImgs,gtLabelImgs);
        }
        	else //RGB case
        	{
//        	     perform pose estimation based on forest output, poses are stored in RANSAC class
        	    ransacTime = ransac2D.estimatePose(
        		gtSegImgs,gtLabelImgs,
        		bb3Ds);
        	}
        
        std::cout << BLUETEXT("RANSAC time: " << ransacTime << "ms") << std::endl << std::endl;
        avgRansacTime += ransacTime;
        
        
        
        //	// do evaluation per object
        //	if(gp->fP.useDepth) // RGB-D case
        //	{
        //	    evalObjectsComplete<Ransac3D>(
        //		i,
        //		objEval,
        //		ransac3D.poses,
        //		rForests[forestPasses-1],
        //		leafImgs,
        //		infos,
        //		trainingInfos,
        //		rProbabilities,
        //		gtObjImgs,
        //		gtSegImgs,
        //		bb3Ds,
        //		pointClouds,
        //		poseThresholdIoU,
        //		poseThresholdHS,
        //		poseThresholdTrans,
        //		poseThresholdRot
        //	    );
        //	}
        //	else // RGB case
        //	{
        //	    evalObjectsComplete<Ransac2D>(
        //		i,
        //		objEval,
        //		ransac2D.poses,
        //		rForests[forestPasses-1],
        //		leafImgs,
        //		infos,
        //		trainingInfos,
        //		rProbabilities,
        //		gtObjImgs,
        //		gtSegImgs,
        //		bb3Ds,
        //		pointClouds,
        //		poseThresholdIoU,
        //		poseThresholdHS,
        //		poseThresholdTrans,
        //		poseThresholdRot
        //	    );
        //	}
        
        // display result and intermediate outputs
        if(gp->tP.displayWhileTesting)
        {
            // draw ground truth poses
            jp::img_bgr_t gtPoses = colorData.clone();
            drawBBs(gtPoses, infos, bb3Ds, colors);
            cv::imshow("Ground Truth Poses", gtPoses);
            
            // draw estimated poses (as 3D bounding boxes)
            if(gp->fP.useDepth) //RGB-D case
            {
                int ct=0;
                for(const auto& entry : ransac3D.poses){ // iterate through all poses found by RANSAC
                    if(ct==0){
                        drawBB(colorData, entry.second.pose, bb3Ds[entry.second.objID-1], colors[entry.second.objID-1]);
                        ct=ct+1;
                    }
                }
            }
            	    else //RGB case
            	    {
            		for(const auto& entry : ransac2D.poses) // iterate through all poses found by RANSAC
            		    drawBB(colorData, entry.second.pose, bb3Ds[entry.second.objID-1], colors[entry.second.objID-1]);
            	    }
            cv::imshow("Estimated Poses", colorData);
            
            // draw estimated segmentation and object coordinates (combined in one image)
            //	    jp::img_bgr_t estSegImg = jp::img_bgr_t::zeros(colorData.rows, colorData.cols);
            //	    jp::img_bgr_t estObjImg = jp::img_bgr_t::zeros(colorData.rows, colorData.cols);
            //	    drawForestEstimation(estSegImg, estObjImg, rForests[forestPasses-1], leafImgs, infos, rProbabilities, colors);
            //	    cv::imshow("Estimated Object Segmentation", estSegImg);
            //	    cv::imshow("Estimated Object Coordinates", estObjImg);
            //
            //	    // draw ground truth segmentation and object coordinates (combined in one image)
            //	    jp::img_bgr_t gtSegImg = jp::img_bgr_t::zeros(colorData.rows, colorData.cols);
            //	    jp::img_bgr_t gtObjImg = jp::img_bgr_t::zeros(colorData.rows, colorData.cols);	    
            //	    drawGroundTruth(gtSegImg, gtObjImg, gtObjImgs, gtSegImgs, infos, colors);
            //	    cv::imshow("Ground Truth Object Segmentation", gtSegImg);
            //	    cv::imshow("Ground Truth Object Coordinates", gtObjImg);
            //        cv::imshow("Ground Truth Poses", gtLabelImgs[0]);
            
            
            
            cv::waitKey();
        }
        
        imageCount++;
    }
    
    // store evaluation results in file
    if(!gp->tP.displayWhileTesting)
        storeEval(objEval, avgForestTime / imageCount, avgRansacTime / imageCount);
    
    // print evaluation results to console
    printEval(objEval);
    return 0;
}

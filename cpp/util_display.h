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
#include "properties.h"
#include "regression_tree.h"
#include "combined_feature.h"

#include "Hypothesis.h"

/** Utility functions for displaying custom image types and for vizualization of results. */

/**
 * @brief Convert a object coordinate image to an BGR image.
 * 
 * Object coordinates are mapped to the RGB cube. Min and max values of object 
 * coordinate are taken from the info file (object extent).
 * 
 * @param img Object coordinate image to display.
 * @param info Info type of the object the object coordinates belong to.
 * @return cv::Mat 8bit BGR vizualization of the object coordinates.
 */
cv::Mat convertForDisplay(const jp::img_coord_t& img, jp::info_t info)
{
    cv::Mat result(img.size(), CV_8UC3);
	
    for(int x = 0; x < img.cols; x++){
        for(int y = 0; y < img.rows; y++){
            for(int channel = 0; channel < 3; channel++)
    {
	float maxExtent = info.extent(channel) * 1000.f; // in meters
        float aaa= img(y, x)(channel);
//        std::cout<<"aaa="<<aaa<<std::endl;
	int coord = (int) (img(y, x)(channel) + maxExtent / 2.f); // shift zero point so all values are positive
	result.at<cv::Vec3b>(y, x)[channel] = (uchar) ((coord / maxExtent) * 255); // rescale to RGB range
    }
            }
        }
    
    return result;
}

/**
 * @brief Convert a depth image to a grayscale image.
 * 
 * The method determines minimal and maximal depth of the given image and maps this range to grayscale.
 * 
 * @param img Depth image to display.
 * @return cv::Mat 8bit grayscale vizualization of the object coordinates.
 */
cv::Mat convertForDisplay(const jp::img_depth_t& img)
{
    //determine min and max depth first
    jp::depth_t minDepth = 10000;
    jp::depth_t maxDepth = 0;
    
    for(int x = 0; x < img.cols; x++)
    for(int y = 0; y < img.rows; y++)
    {
	maxDepth = std::max(maxDepth, img(y, x));
	if(img(y, x) > 0)  //skip pixels without depth
	    minDepth = std::min(minDepth, img(y, x));
    }
    
    cv::Mat result(img.size(), CV_8U);

    // map depth values to gray value using the min max range 
    for(int x = 0; x < img.cols; x++)
    for(int y = 0; y < img.rows; y++)
    {
	if(img(y, x) == 0)
	    result.at<uchar>(y, x) = 0; // depth holes are mapped to black
	else
	    result.at<uchar>(y, x) = 255 - (uchar) (((img(y, x) - minDepth) / (float) (maxDepth - minDepth)) * 255);
    }

    return result;
}

/**
 * @brief Draws the object bounding box using the given pose.
 * 
 * @param img Input image.
 * @param trans Object pose.
 * @param bb3D The 3D object bounding box.
 * @param color Color of the bounding box to draw.
 * @return void
 */
void drawBB(jp::img_bgr_t& img, const jp::cv_trans_t& trans, const std::vector<cv::Point3f>& bb3D, const cv::Scalar& color)
{
    std::vector<cv::Point2f> bb2D;
    
    // project 3D bounding box vertices into the image
    cv::Mat_<float> camMat = GlobalProperties::getInstance()->getCamMat();
    cv::projectPoints(bb3D, trans.first, trans.second, camMat, cv::Mat(), bb2D);

    int lineW = 2; // line width
    int lineType = CV_AA; //anti aliasing
    
    // draw sligthly bigger black line underneath the colored line for a nice halo effect
    cv::line(img, bb2D[0], bb2D[1], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[1], bb2D[3], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[3], bb2D[2], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[2], bb2D[0], cv::Scalar(0, 0, 0), lineW+2, lineType);

    cv::line(img, bb2D[2], bb2D[6], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[0], bb2D[4], cv::Scalar(0, 0, 0), lineW+2, lineType);    
    cv::line(img, bb2D[1], bb2D[5], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[3], bb2D[7], cv::Scalar(0, 0, 0), lineW+2, lineType);
    
    cv::line(img, bb2D[4], bb2D[5], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[5], bb2D[7], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[7], bb2D[6], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[6], bb2D[4], cv::Scalar(0, 0, 0), lineW+2, lineType);
    
    // draw colored line
    cv::line(img, bb2D[0], bb2D[1], color, lineW, lineType);
    cv::line(img, bb2D[1], bb2D[3], color, lineW, lineType);
    cv::line(img, bb2D[3], bb2D[2], color, lineW, lineType);
    cv::line(img, bb2D[2], bb2D[0], color, lineW, lineType);
    
    cv::line(img, bb2D[2], bb2D[6], color, lineW, lineType);
    cv::line(img, bb2D[0], bb2D[4], color, lineW, lineType);
    cv::line(img, bb2D[1], bb2D[5], color, lineW, lineType);
    cv::line(img, bb2D[3], bb2D[7], color, lineW, lineType);
    
    cv::line(img, bb2D[4], bb2D[5], color, lineW, lineType);
    cv::line(img, bb2D[5], bb2D[7], color, lineW, lineType);
    cv::line(img, bb2D[7], bb2D[6], color, lineW, lineType);
    cv::line(img, bb2D[6], bb2D[4], color, lineW, lineType);
}

/**
 * @brief Draw a list of object bounding boxes using ground truth information.
 * 
 * The method skips objects not visible according to the respective info file.
 * 
 * @param img Input image.
 * @param infos Ground truth information about the objects. Contains pose and visibility information.
 * @param bb3Ds The 3D object bounding boxes. Same count as infos.
 * @param colors Colors of the bounding box to draw. Same count as infos.
 * @return void
 */
void drawBBs(
    jp::img_bgr_t& img, 
    const std::vector<jp::info_t>& infos, 
    const std::vector<std::vector<cv::Point3f>>& bb3Ds, 
    const std::vector<cv::Scalar>& colors)
{
    for(unsigned o = 0; o < infos.size(); o++)
    {
	if(!infos[o].visible) continue;
	drawBB(img, jp::our2cv(infos[o]), bb3Ds[o], colors[o]);
    }  
}

/**
 * @brief Vizualize object coordinate prediction of one tree for a certain object.
 * 
 * The method extracts for each pixel the dominant object coordinate 
 * prediction (mode with most support) of one tree.
 * 
 * @param forest Random forest that made the prediction.
 * @param leafImgs Prediction in the form of leaf indices per pixel. One leaf image for each tree in the forest.
 * @param info Info of the object the prediction was made for.
 * @param treeIdx Index of tree to vizualize. the prediction of.
 * @param objID Object ID of the object the prediction should be vizualized for.
 * @return cv::Mat BGR vizualization of the object coordinate prediction.
 */
cv::Mat getModeImg(
    const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
    const std::vector<jp::img_leaf_t>& leafImgs,
    jp::info_t info, int treeIdx, jp::id_t objID)
{
    if(leafImgs.empty()) return cv::Mat();
  
    treeIdx = std::max(0, treeIdx); // use default tree if treeIdx < 0
    
    jp::img_coord_t modeImg(leafImgs[0].rows, leafImgs[0].cols);
  
    for(int x = 0; x < modeImg.cols; x++)
    for(int y = 0; y < modeImg.rows; y++)
    {
	size_t leaf = leafImgs[treeIdx](y, x);
	const std::vector<jp::mode_t>* modes = forest[treeIdx].getModes(leaf, objID);
	
	if(modes->empty() || !leaf)
	    modeImg(y, x) = jp::coord3_t(0, 0, 0); // No distribution stored in leaf.
	else
	    modeImg(y, x) = modes->at(0).mean;
    }
    
    return convertForDisplay(modeImg, info);
}

/**
 * @brief Vizualizes the prediction (segmentation and object coordinates) of the forest for all objects.
 * 
 * The method will create a combined vizualization of the forest prediction for multiple objects. 
 * Prediction of different objects a blended according to the object probabilities predicted for 
 * each pixel. Segmentations of seperate objects are displayed with different colors. The method 
 * with use the prediction of one tree only and the dominant moder per pixel (largest support) 
 * for the vizualition of the object coordinates.
 * 
 * @param segImg Output parameter, soft segmentation (or probability prediction), merged for all objects.
 * @param objImg Output parameter, object coordinate prediction, merged for all objects.
 * @param forest Random forest that made the prediction.
 * @param leafImgs Prediction in the form of leaf indices per pixel. One leaf image for each tree in the forest.
 * @param infos Infos about the objects the prediction was made for. One for each object.
 * @param probabilities Probability predictions of the forest. One for each object.
 * @param colors Colors to use for each object. One for each object.
 * @return void
 */
void drawForestEstimation(
    jp::img_bgr_t& segImg,
    jp::img_bgr_t& objImg,
    const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
    const std::vector<jp::img_leaf_t>& leafImgs,
    const std::vector<jp::info_t>& infos,
    const std::vector<jp::img_stat_t>& probabilities,
    const std::vector<cv::Scalar>& colors)
{
    std::vector<jp::img_bgr_t> estObjImgs;
    
    // get separate object coordinate vizualizations for each obejct.
    for(unsigned o = 0; o < infos.size(); o++)
	estObjImgs.push_back(getModeImg(forest, leafImgs, infos[o], 0, o+1));
    
    // blend predictions of all objects
    #pragma omp parallel for
    for(unsigned x = 0; x < segImg.cols; x++)
    for(unsigned y = 0; y < segImg.rows; y++)
    for(unsigned p = 0; p < probabilities.size(); p++)
    {
	float prob = probabilities[p](y, x); // object probability is used for blending the predictions
			
	segImg(y, x)[0] += prob * colors[p][0];
	segImg(y, x)[1] += prob * colors[p][1];
	segImg(y, x)[2] += prob * colors[p][2];
	
	objImg(y, x)[0] += prob * estObjImgs[p](y, x)[0];
	objImg(y, x)[1] += prob * estObjImgs[p](y, x)[1];
	objImg(y, x)[2] += prob * estObjImgs[p](y, x)[2];
    }  
}

/**
 * @brief Creates a combined vizualization of ground truth object segmentations and object coordinates.
 * 
 * @param segImg Output parameter. Combined segmentation masks of all objects. Each object segmentation uses a different color.
 * @param objImg Output parameter. Combined object coordinates ground truth of all objects.
 * @param gtObjImgs Individual object coordinate ground truth images. One per object.
 * @param gtSegImgs Individual object segmentation masks. One per object.
 * @param infos Infos about the objects to do the vizualization for. One per object.
 * @param colors Object colors to use for the vizualization. One per object.
 * @return void
 */
void drawGroundTruth(
    jp::img_bgr_t& segImg,
    jp::img_bgr_t& objImg,
    const std::vector<jp::img_coord_t>& gtObjImgs,
    const std::vector<jp::img_id_t>& gtSegImgs,
    const std::vector<jp::info_t>& infos,
    const std::vector<cv::Scalar>& colors)
{
    for(unsigned o = 0; o < infos.size(); o++)
    {
	if(!infos[o].visible) continue; // skip invisible objects
        
        jp::img_bgr_t curObjImg = convertForDisplay(gtObjImgs[o], infos[o]);
        	#pragma omp parallel for
	for(unsigned x = 0; x < segImg.cols; x++)
	for(unsigned y = 0; y < segImg.rows; y++)
	{
	    if(!gtSegImgs[o](y, x)) continue; // skip all pixels that do not belong to the object
			    
	    segImg(y, x)[0] = colors[o][0];
	    segImg(y, x)[1] = colors[o][1];
	    segImg(y, x)[2] = colors[o][2];
	    
	    objImg(y, x)[0] = curObjImg(y, x)[0];
	    objImg(y, x)[1] = curObjImg(y, x)[1];
	    objImg(y, x)[2] = curObjImg(y, x)[2];
	}
    }
}

/**
 * @brief Creates a list of colors to use for vizualization.
 * 
 * @return std::vector< cv::Scalar, std::allocator< void > > List of colors.
 */
std::vector<cv::Scalar> getColors()
{
    std::vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar(255, 0, 0));
    colors.push_back(cv::Scalar(0, 255, 0));
    colors.push_back(cv::Scalar(0, 0, 255));
    
    colors.push_back(cv::Scalar(255, 255, 0));
    colors.push_back(cv::Scalar(255, 0, 255));
    colors.push_back(cv::Scalar(0, 255, 255));
    
    colors.push_back(cv::Scalar(255, 127, 0));
    colors.push_back(cv::Scalar(255, 0, 127));
    colors.push_back(cv::Scalar(255, 127, 127));
    
    colors.push_back(cv::Scalar(127, 255, 0));
    colors.push_back(cv::Scalar(0, 255, 127));
    colors.push_back(cv::Scalar(127, 255, 127));

    colors.push_back(cv::Scalar(127, 0, 255));
    colors.push_back(cv::Scalar(0, 127, 255));
    colors.push_back(cv::Scalar(127, 127, 255));
    
    for(unsigned c = colors.size(); c < 50; c++)
	colors.push_back(cv::Scalar(127, 127, 127));
    
    return colors;
}

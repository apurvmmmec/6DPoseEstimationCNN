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
#include "thread_rand.h"
#include "properties.h"
#include "util.h"
#include "training_samples.h"

/** Utility functions used in all features or for sampling features. */

namespace jp
{
    /** Container for pixel positions related to features */
    struct FeaturePoints
    {
	FeaturePoints()
	{
	  xc = yc = x1 = y1 = x2 = y2 = 0;
	}
      
	int xc, yc; // Absolute x,y of feature center.
	int x1, y1; // Absolute x,y of feature offset vector 1 (scaled and potentially rotated).
	int x2, y2; // Absolute x,y of feature offset vector 2 (scaled and potentially rotated).
    };
    
    /**
     * @brief Calculates absolute feature offset positions.
     * 
     * Scales (and optionally rotates) the two offset vectors and clamps all absolute positions at the image border.
     * 
     * @param x Feature center x in pixels.
     * @param y Feature center y in pixels.
     * @param off1_x Relative feature offset 1 in x direction in pixels.
     * @param off1_y Relative feature offset 1 in y direction in pixels.
     * @param off2_x Relative feature offset 2 in x direction in pixels.
     * @param off2_y Relative feature offset 1 in y direction in pixels.
     * @param scale Factor with which the patch should be scaled before feature calculation (1.0 means no scaling).
     * @param width Image width in pixels.
     * @param height Image height in pixels.
     * @return jp::FeaturePoints Clamped absolute positions of feature center and offsets.
     */
    inline FeaturePoints getFeaturePoints(
	int x, int y, int off1_x, int off1_y, int off2_x, int off2_y, 
	float scale, int width, int height)
    {
	FeaturePoints featurePoints;
	GlobalProperties* gp = GlobalProperties::getInstance();      
	
	// ensure position is within image border
	x = clamp(x, 0, width - 1);
	y = clamp(y, 0, height - 1);
	
	featurePoints.xc = x;
	featurePoints.yc = y;

	// apply scale, scaling the patch is approximated by scaling the offset vectors
	float x1 = off1_x * scale;
	float y1 = off1_y * scale;
	float x2 = off2_x * scale;
	float y2 = off2_y * scale;	
	
	// relative offsets to absolute positions
	if(gp->fP.training && !gp->rotations.empty())
	{
	    // approximate random patch rotation by rotating offset vectors
	    int r = irand(0, gp->rotations.size());
	  
	    featurePoints.x1 = x + (gp->rotations[r].at<double>(0, 0) * x1 + gp->rotations[r].at<double>(0, 1) * y1 + gp->rotations[r].at<double>(0, 2));
	    featurePoints.y1 = y + (gp->rotations[r].at<double>(1, 0) * x1 + gp->rotations[r].at<double>(1, 1) * y1 + gp->rotations[r].at<double>(1, 2));
	    featurePoints.x2 = x + (gp->rotations[r].at<double>(0, 0) * x2 + gp->rotations[r].at<double>(0, 1) * y2 + gp->rotations[r].at<double>(0, 2));
	    featurePoints.y2 = y + (gp->rotations[r].at<double>(1, 0) * x2 + gp->rotations[r].at<double>(1, 1) * y2 + gp->rotations[r].at<double>(1, 2));
	}
	else
	{
	    featurePoints.x1 = x + x1;
	    featurePoints.y1 = y + y1;
	    featurePoints.x2 = x + x2;
	    featurePoints.y2 = y + y2;
	}
	
	// ensure offsets are within image border
	featurePoints.x1 = clamp(featurePoints.x1, 0, width - 1);
	featurePoints.y1 = clamp(featurePoints.y1, 0, height - 1);
	featurePoints.x2 = clamp(featurePoints.x2, 0, width - 1);
	featurePoints.y2 = clamp(featurePoints.y2, 0, height - 1);
	
	return featurePoints;
    }
    
    /**
     * @brief Calculates absolute feature offset position.
     * 
     * Scales (and optionally rotates) the offset vector and clamps all absolute positions at the 
     * image border. Information about the second offset vector is set to zero.
     * 
     * @param x Feature center x in pixels.
     * @param y Feature center y in pixels.
     * @param off1_x Relative feature offset 1 in x direction in pixels.
     * @param off1_y Relative feature offset 1 in y direction in pixels.
     * @param scale Factor with which the patch should be scaled before feature calculation (1.0 means no scaling).
     * @param width Image width in pixels.
     * @param height Image height in pixels.
     * @return jp::FeaturePoints Clamped absolute positions of feature center and offset.
     */
    inline FeaturePoints getFeaturePoints(
	int x, int y, int off1_x, int off1_y, 
	float scale, int width, int height)
    {
	FeaturePoints featurePoints;
	GlobalProperties* gp = GlobalProperties::getInstance();      
	
	// ensure position is within image border
	x = clamp(x, 0, width - 1);
	y = clamp(y, 0, height - 1);
	
	featurePoints.xc = x;
	featurePoints.yc = y;

	// apply scale, scaling the patch is approximated by scaling the offset vectors
	float x1 = off1_x * scale;
	float y1 = off1_y * scale;
	
	// relative offsets to absolute positions
	if(gp->fP.training && !gp->rotations.empty())
	{
	    // apply random patch rotation
	    int r = irand(0, gp->rotations.size());
	    
	    featurePoints.x1 = x + (gp->rotations[r].at<double>(0, 0) * x1 + gp->rotations[r].at<double>(0, 1) * y1 + gp->rotations[r].at<double>(0, 2));
	    featurePoints.y1 = y + (gp->rotations[r].at<double>(1, 0) * x1 + gp->rotations[r].at<double>(1, 1) * y1 + gp->rotations[r].at<double>(1, 2));
	}
	else
	{
	    featurePoints.x1 = x + x1;
	    featurePoints.y1 = y + y1;
	}
	
	// ensure offsets are within image border
	featurePoints.x1 = clamp(featurePoints.x1, 0, width - 1);
	featurePoints.y1 = clamp(featurePoints.y1, 0, height - 1);
	
	return featurePoints;
    }    

}

/**
 * @brief Create a random feature with the feature threshold set as the feature response at a random image pixel.
 * 
 * @param data Image from which the random pixel is drawn.
 * @param sampler Feature sampler used to create a random feature.
 * @return TFeatureSampler::feature_t A random feature.
 */
template<typename TFeatureSampler>
typename TFeatureSampler::feature_t sampleFromRandomPixel(
    const jp::img_data_t& data,
    const TFeatureSampler& sampler)
{
    // create a list of all pixel locations covered by the object
    std::vector<sample_t> objPixels;
    objPixels.reserve(data.seg.cols * data.seg.rows);

    for(unsigned y = 0; y < data.seg.rows; y++)
    for(unsigned x = 0; x < data.seg.cols; x++)
    {
	if(data.seg(y, x)) objPixels.push_back(sample_t(x, y));
    }

    //set the threshold of the new feature to the response of 1 random pixel
    int pixelIdx = irand(0, objPixels.size());

    // Create random feature, but then set the threshold by sampling from the data
    typename TFeatureSampler::feature_t feat = sampler.sampleFeature();
    feat.setThreshold(feat.computeResponse(objPixels[pixelIdx].x, objPixels[pixelIdx].y, objPixels[pixelIdx].scale, data));

    return feat;
}

/**
 * @brief Create multiple random features with the feature thresholds set as the feature response at random image pixels of randomly selected images.
 * 
 * @param data Images from which the random pixels are drawn.
 * @param count How many features to create.
 * @param sampler Feature sampler used to create random features.
 * @return std::vector< TFeatureSampler::feature_t> A list of random features.
 */
template<typename TFeatureSampler>
std::vector<typename TFeatureSampler::feature_t> sampleFromRandomPixels(
    const std::vector<jp::img_data_t>& data,
    unsigned int count, const TFeatureSampler& sampler)
{
    std::vector<size_t> imageIdx(count);
    for (unsigned int i = 0; i < count; ++i)
	imageIdx[i] = irand(0, data.size());
    std::sort(imageIdx.begin(), imageIdx.end());

    // Sample feature tests
    std::vector<typename TFeatureSampler::feature_t> rv(count);
    for (unsigned int i = 0; i < count; ++i)
	rv[i] = sampleFromRandomPixel(data[imageIdx[i]], sampler);

    return rv;
}
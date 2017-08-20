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

#include "training_samples.h"

#include "../core/properties.h"
#include "../core/sampler2D.h"

void samplePixels(const jp::img_id_t& segmentation, const jp::info_t& info, std::vector<sample_t>& samples, const jp::img_depth_t* depth)
{
    // create a list of all valid pixel positions
    std::vector<sample_t> pxValid;
    pxValid.reserve(segmentation.cols * segmentation.rows);
        
    for(unsigned x = 0; x < segmentation.cols; x++)
    for(unsigned y = 0; y < segmentation.rows; y++)
    {
	if(segmentation(y, x)) pxValid.push_back(sample_t(x, y));
    }

    // parameters used for random scale sampling
    GlobalProperties* gp = GlobalProperties::getInstance();
    float inputDepth = -info.center(2);
    float scaleMax = gp->fP.scaleMax;
    float scaleMin = gp->fP.scaleMin;

    for (size_t ci = 0; ci < samples.size(); ++ci)
    {
        // sample a random pixel
	sample_t sample = pxValid[irand(0, pxValid.size())];
	
	// set scale 
	if(depth == NULL || depth->operator()(sample.y, sample.x) == 0)
	{
	    // no depth channel available, sample a random scale
	    sample.scale = drand(scaleMin, scaleMax);
	    
	    if(gp->fP.scaleRel) // scale patch relative to the distance in the training image, i.e. normalize the distance in the training image before sccaling
		sample.scale = sample.scale / inputDepth;
	}
	else
	{
	    // scale patch according to available depth channel
	    sample.scale = 1.f / depth->operator()(sample.y, sample.x) * 1000.f;
	}
	
	samples[ci] = sample;
    }
}

void samplePixels(const jp::img_stat_t& prob, const jp::info_t& info, std::vector<sample_t>& samples, const jp::img_depth_t* depth)
{
    // create a sampler to draw pixels according to probability map (= soft segmentation)
    Sampler2D sampler(prob);
    
    // parameters used for random scale sampling
    GlobalProperties* gp = GlobalProperties::getInstance();
    float inputDepth = -info.center(2);
    float scaleMax = gp->fP.scaleMax;
    float scaleMin = gp->fP.scaleMin;
    
    // sample random size    
    for (size_t ci = 0; ci < samples.size(); ++ci)
    {
	// sample a random pixel according to pixel probabilities
	sample_t sample;
	cv::Point2f pt = sampler.drawInRect(cv::Rect(cv::Point(0, 0), cv::Point(prob.cols-1, prob.rows-1)));
	sample.x = pt.x;
	sample.y = pt.y;
	
	if(depth == NULL || depth->operator()(sample.y, sample.x) == 0)
	{
	    // no depth channel available, sample a random scale
	    sample.scale = drand(scaleMin, scaleMax);
	    
	    if(gp->fP.scaleRel) // scale patch relative to the distance in the training image, i.e. normalize the distance in the training image before sccaling
		sample.scale = sample.scale / inputDepth;
	}
	else
	{
	    // scale patch according to available depth channel
	    sample.scale = 1.f / depth->operator()(sample.y, sample.x) * 1000.f;
	}
	
	samples[ci] = sample;
    }
}
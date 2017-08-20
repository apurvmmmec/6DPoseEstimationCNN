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

#include "../core/types.h"

/** Methods for drawing training samples from images. */

/**
 * @brief A training sample (image patch).
 */
struct sample_t 
{
    int x, y; // position of the center pixel
    float scale; // patch scale factor 

    sample_t() : x(0), y(0), scale(1) {}
    sample_t(int x, int y) : x(x), y(y), scale(1) {}
    sample_t(int x, int y, float scale) : x(x), y(y), scale(scale) {}
};

/**
 * @brief Generates a list of samples.
 * 
 * Generates a list of samples using a segmentation mask. Samples are only 
 * draw from within the segmentation. Samples are scaled according to the 
 * depth channel if supplied. If depth is NULL then samples are scaled
 * randomly according to the strategy set in the properties.
 * 
 * @param segmentation Binary segmentation mask. Pixels where segmentation is 0 will not be drawn.
 * @param info Ground truth pose annotation for the given image. Used to scale patches randomly.
 * @param samples Output paramter. List of samples for this image.
 * @param depth Optional parameter. Used to scale samples. Random scales will be used if depth is NULL.
 * @return void
 */
void samplePixels(const jp::img_id_t& segmentation, const jp::info_t& info, std::vector<sample_t>& samples, const jp::img_depth_t* depth);

/**
 * @brief Generates a list of samples.
 * 
 * Generates a list of samples using a soft segmentation mask. Samples 
 * drawn according to the probability of each pixel. Samples are scaled 
 * according to the depth channel if supplied. If depth is NULL then 
 * samples are scaled randomly according to the strategy set in the properties.
 * 
 * @param segmentation Soft segmentation mask. Pixels are drawn according to the probability of each pixel.
 * @param info Ground truth pose annotation for the given image. Used to scale patches randomly.
 * @param samples Output paramter. List of samples for this image.
 * @param depth Optional parameter. Used to scale samples. Random scales will be used if depth is NULL.
 * @return void
 */
void samplePixels(const jp::img_stat_t& prob, const jp::info_t& info, std::vector<sample_t>& samples, const jp::img_depth_t* depth);
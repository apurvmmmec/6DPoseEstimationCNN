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

#include "dataset.h"

namespace jp
{
  /**
   * @brief This class bundles multiple datasets (for multiple objects) and offers a view as if it was one big dataset (concatination of datasets).
   */
  class MultiDataset 
    {
    public:
	
        /**
         * @brief Constructor. 
         * 
         * @param datasets Lists of datasets.
         */
        MultiDataset(std::vector<Dataset> datasets) : datasets(datasets)
	{
	    // build dataset mapping (accumulated size to dataset index)
	    idxTable[0] = std::pair<size_t, size_t>(0, 0);
	    sizeSum = 0;
	    size_t lastSum = 0;

	    for(int i = 0; i < datasets.size(); i++)
	    {
		sizeSum += datasets[i].size();
		idxTable[sizeSum - 1] = std::pair<size_t, size_t>(lastSum, i);
		lastSum = sizeSum;
	    }

	    std::cout << "Training images found: " << sizeSum << std::endl;
	}

	/**
	 * @brief Returns the size (number of frames) of the combined dataset.
	 * 
	 * @return size_t Number of frames in the dataset.
	 */
	size_t size() const 
	{ 
	    return sizeSum; 
	}

	/**
	 * @brief Size of the dataset of a specific object.
	 * 
	 * @param objID Query object ID.
	 * @return size_t Number of frames for this object.
	 */
	size_t size(jp::id_t objID) const 
	{ 
	    return datasets[objID - 1].size(); 
	}

	/**
	 * @brief Get the object that the given frame belongs to.
	 * 
	 * @param i Frame number.
	 * @return jp::id_t Object ID.
	 */
	jp::id_t getObjID(size_t i) const
	{
	    return datasets.at(getDB(i)).getObjID();
	}
	
	/**
	 * @brief Get the file name (RGB image) of the given frame number.
	 * 
	 * @param i Frame number.
	 * @return std::string File name.
	 */
	std::string getFileName(size_t i) const
	{
	    return datasets.at(getDB(i)).getFileName(getIdx(i));
	}
	
	/**
	 * @brief Get the RGB image of the given frame.
	 * 
	 * @param i Frame number.
	 * @param img Output parameter. RGB image.
	 * @param noseg If true, image will not be segmented.
	 * @return void
	 */
	void getBGR(size_t i, jp::img_bgr_t& img, bool noseg) const
	{
	    datasets.at(getDB(i)).getBGR(getIdx(i), img, noseg);
	}
	
	/**
	 * @brief Get the depth image of the given frame.
	 * 
	 * @param i Frame number.
	 * @param img Output parameter. depth image.
	 * @param noseg If true, image will not be segmented.
	 * @return void
	 */	
	void getDepth(size_t i, jp::img_depth_t& img, bool noseg) const
	{
	    datasets.at(getDB(i)).getDepth(getIdx(i), img, noseg);
	}
	
	/**
	 * @brief Get the RGB-D image of the given frame.
	 * 
	 * @param i Frame number.
	 * @param img Output parameter. RGB-D image.
	 * @param noseg If true, image will not be segmented.
	 * @return void
	 */
	void getBGRD(size_t i, jp::img_bgrd_t& img, bool noseg) const
	{
	    datasets.at(getDB(i)).getBGRD(getIdx(i), img, noseg);
	}	
	
	/**
	 * @brief Get the ground truth segmentation mask of the given frame.
	 * 
	 * @param i Frame number.
	 * @param img Output parameter. Segmentation mask.
	 * @return void
	 */	
	void getSegmentation(size_t i, jp::img_id_t& seg) const
	{
	    datasets.at(getDB(i)).getSegmentation(getIdx(i), seg);
	}
	
	/**
	 * @brief Get the ground truth object coordinate image of the given frame.
	 * 
	 * @param i Frame number.
	 * @param img Output parameter. Object coordinate image.
	 * @return void
	 */
	void getObj(size_t i, jp::img_coord_t& img) const
	{
	    return datasets.at(getDB(i)).getObj(getIdx(i), img);
	}
	
	/**
	 * @brief Get the camera coordinate image of the given frame (generated from the depth channel).
	 * 
	 * @param i Frame number.
	 * @param img Output parameter. Camera coordinate image.
	 * @return void
	 */	
	void getEye(size_t i, jp::img_coord_t& img) const
	{
	    return datasets.at(getDB(i)).getEye(getIdx(i), img);
	}

	/**
	 * @brief Get the ground truth annotations of the given frame.
	 * 
	 * @param i Frame number.
	 * @param info Output parameter. Ground truth.
	 * @return void
	 */		
	void getInfo(size_t i, jp::info_t& info) const
	{
	    datasets.at(getDB(i)).getInfo(getIdx(i), info);
	}

    private:
      
        /**
         * @brief Returns the index of the dataset that the given frame number falls into.
         * 
	 * @param i Frame number.
         * @return size_t Dataset index.
         */
        size_t getDB(size_t i) const
	{
	    return (*idxTable.lower_bound(i)).second.second;
	}

	/**
	 * @brief Returns the frame number for the object dataset the given (global) frame number falls into.
	 * 
	 * @param i Frame number.
	 * @return size_t Frame number of the associated object dataset.
	 */
	size_t getIdx(size_t i) const
	{
	    return i - (*idxTable.lower_bound(i)).second.first;
	}

	std::vector<Dataset> datasets; // list of object datasets

	// maps combined indices to datasets and their indices
	std::map<size_t, std::pair<size_t, size_t>> idxTable; 
	size_t sizeSum;
    };
}
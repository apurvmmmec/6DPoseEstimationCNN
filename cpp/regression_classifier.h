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

#include "regression_tree.h"
#include "combined_feature.h"

namespace jp
{
    /**
     * @brief Class for duing predictions using a regression forest.
     */
    template <typename TFeature>
    class RegressionForestClassifier
    {
    public:
        /**
         * @brief Constructor.
         * 
         * @param forest Regression forest to do predictions with.
         */
        RegressionForestClassifier(std::vector<RegressionTree<TFeature>>& forest) : forest(forest) {}
    
        /**
         * @brief Feeds an image through each tree patch by patch and stores the leaf index.
         *
	 * @param data The input image (container of all channels the forest features work on).
	 * @param leafImgs Output parameter. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
	 * @param depth Optional. Used for scaling features if not NULL. 
	 * 
         * @return void
         */
        void classify(const jp::img_data_t& data, std::vector<img_leaf_t>& leafImgs, const jp::img_depth_t* depth)
	{
	    if(leafImgs.size() != forest.size())
		leafImgs.resize(forest.size());
	  
	    for(unsigned t = 0; t < forest.size(); t++)
		forest[t].structure.getLeafImg(data, leafImgs[t], depth);
	}

	/**
	 * @brief Reads out the probability for a specific object at a specific pixel of a single tree in the forest.
	 * 
	 * @param forest Regression forest that made the prediction.
	 * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
	 * @param treeIdx Index of the tree the probability should be looked up in.
	 * @param objID The ID of the object of which the probability should be looked up.
	 * @param x X pixel position for with the probability should be looked up.
	 * @param y Y pixel position for with the probability should be looked up.
	 * @return float Object probability.
	 */
	inline float getProbability(
	    const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	    const std::vector<jp::img_leaf_t>& leafImgs,
	    unsigned treeIdx,
	    jp::id_t objID,
	    unsigned x,
	    unsigned y) const
	{
	    size_t leaf = leafImgs[treeIdx](y, x); // which tree leaf corresponds to the query pixel?
	    return forest[treeIdx].getObjPixels(leaf, objID) / (forest[treeIdx].getLeafPixels(leaf) + 1.f); // calculate probability from sample frequencies, +1 in the denominator for a robust estimate
	}

	/**
	 * @brief Calculates probability maps for all objects given an forest prediction.
	 * 
	 * Probabilities are calculated by multiplying the sample frequencies of individual
	 * trees and normalizing. This gives more contrasted predictions than the usual 
	 * averaging of sample frequencies.
	 * 
	 * @param forest Random forest that did the prediction.
	 * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
	 * @param probs Output parameter. One image per object. Each pixel stores the probabilitiy to belong to that object (soft segmentation).
	 * @return void
	 */
	void getObjsProb(
	    const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	    const std::vector<jp::img_leaf_t>& leafImgs, 
	    std::vector<jp::img_stat_t>& probs) const
	{
	    int objectCount = GlobalProperties::getInstance()->fP.objectCount;

	    //init probability images
	    for(jp::id_t objID = 1; objID <= objectCount; objID++)
		probs[objID - 1] = jp::img_stat_t::ones(leafImgs[0].rows, leafImgs[0].cols);
	    jp::img_stat_t sumProbs = jp::img_stat_t::ones(leafImgs[0].rows, leafImgs[0].cols);
	    
	    #pragma omp parallel for
	    for(unsigned x = 0; x < leafImgs[0].cols; x++)
	    for(unsigned y = 0; y < leafImgs[0].rows; y++)
	    {
		bool allZero = true; //if all leaf indices are zero its most likely (sic!) that this pixel is undefined 
	 
		// accumulate probs for background
		for(unsigned treeIdx = 0; treeIdx < forest.size(); treeIdx++)
		    sumProbs(y, x) *= getProbability(forest, leafImgs, treeIdx, objectCount + 1, x, y);
	 
		// accumulate probs for objects
		for(jp::id_t objID = 1; objID <= objectCount; objID++)
		{
		    for(unsigned treeIdx = 0; treeIdx < forest.size(); treeIdx++)
		    {
			allZero = allZero && (leafImgs[treeIdx](y, x) == 0);
			probs[objID - 1](y, x) *= getProbability(forest, leafImgs, treeIdx, objID, x, y);
		    }
		    sumProbs(y, x) += probs[objID - 1](y, x);
		}

		//early out if undefined pixes
		if(allZero)
		{
		    for(jp::id_t objID = 1; objID <= objectCount; objID++)
			probs[objID - 1](y, x) = 0.f;
		}
		else
		{
		    for(jp::id_t objID = 1; objID <= objectCount; objID++)
			if(sumProbs(y, x) > 0) probs[objID - 1](y, x) /= sumProbs(y, x);
		}
	    }
	}	
	
    private:
      
	std::vector<RegressionTree<TFeature>>& forest; // Random forest to do classifications with.
    };
}
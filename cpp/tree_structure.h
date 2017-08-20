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

#include <vector>
#include <ios>
#include <stack>
#include <list>
#include <queue>
#include <omp.h>

#include "../core/generic_io.h"
#include "training_samples.h"
#include "../core/image_cache.h"

/** Definition of a binary decision tree structure (no leaf information) and how to learn it from data. */

namespace jp
{
  
    /**
     * @brief Decision tree split node. 
     */
    template <class feature_t>
    struct split_t
    {
	split_t() {}
	split_t(const feature_t& t, int left, int right) : left(left), right(right), feature(t) {}

	int left; // pointer to the left child node
	int right; // pointer to the right child node
	feature_t feature; // feature test to decide which child to follow
    };  
  
    /**
     * @brief Binary decision tree structure.
     */    
    template <typename T>
    class TreeStructure
    {
    public:
	TreeStructure() {}

	/**
	 * @brief Returns true if the node is a leaf.
	 */
	bool isLeaf(int node) const 
	{ 
	    return node < 0; 
	}
	
	/**
	 * @brief Maps a child node ID to a leaf node ID
	 */	
	size_t child2Leaf(int child) const 
	{ 
	    return -child - 1; 
	}
	
	/**
	 * @brief Maps a leaf node ID to a child node ID
	 */		
	int leaf2Child(size_t leaf) const 
	{ 
	    return -leaf - 1; 
	}
	
	/**
	 * @brief Returns the total number of nodes (inner nodes + leaf nodes) of the tree.
	 */	
	size_t getNodeCount() const
	{
	    return size() * 2 + 1;
	}

	/**
	 * @brief Returns the number of split nodes (inner nodes) of the tree.
	 */		
	size_t size() const 
	{ 
	    return splits.size(); 
	}

	/**
	 * @brief Returns true if the tree is empty (zero split nodes).
	 */		
	bool empty() const
	{
	    return splits.empty();
	}
	
	/**
	 * @brief Returns the node ID of the root node.
	 */		
	int root() const 
	{
	    return splits.empty() ? -1 : 0; 
	}

	/**
	 * @brief Adds a split node to the tree.
	 * 
	 * @param data Feature test of the new split node.
	 * @return int Returns node ID of the new node.
	 */
	int addNode(const T& data)
	{
	    split_t<T> node(data, -1, -1);
	    splits.push_back(node);
	    return splits.size() - 1;
	}

	/**
	 * @brief Returns the ID of the child node in the given direction. 
	 * 
	 * @param split Node in question.
	 * @param direction True for right child.
	 * 
	 * @return int Node ID of child.
	 */
	int getSplitChild(int split, bool direction) const
	{
	    return direction ? splits[split].right : splits[split].left;
	}

	/**
	 * @brief Sets the child of a node to a given ID.
	 * 
	 * @param split Node to update.
	 * @param direction True for right child.
	 * @param child Node ID of child to set.
	 * 
	 * @return void
	 */
	void setSplitChild(int split, bool direction, int child)
	{
	    (direction ? splits[split].right : splits[split].left) = child;
	}      

	/**
	 * @brief Starting from the root get the leaf for specific input data.
	 * 
	 * @param test Function that takes a feature test and executes it on the input data.
	 * @return size_t Leaf ID.
	 */
	template <typename Test>
	size_t getLeaf(const Test& test) const
	{
	    int i = root();
	    while (!isLeaf(i))
		i = getSplitChild(i, test(splits[i].feature));
	    return child2Leaf(i);
	}

	/**
	 * @brief Calculates the leaf ID for each pixel of an input image.
	 * 
	 * Evaluates the tree for each pixel of the input image.
	 * 
	 * @param data Input data (image).
	 * @param leafs Output parameter. Image that contains for each pixel of the input image the leaf ID. 
	 * @param depth Optional depth channel of the input image. NULL when no depth is available.
	 * 
	 * @return void
	 */
	void getLeafImg(const jp::img_data_t& data, img_leaf_t& leafs, const jp::img_depth_t* depth) const
	{
	    if((leafs.rows != data.seg.rows) || (leafs.cols != data.seg.cols))
		leafs = img_leaf_t(data.seg.rows, data.seg.cols);
	    
	    #pragma omp parallel for
	    for(unsigned x = 0; x < data.seg.cols; x++)
	    for(unsigned y = 0; y < data.seg.rows; y++)
	    {
    		float scale = 1.f;
		if(depth != NULL) // in case a depth channel is given, the patch is scaled according to the depth of the pixel
		{
		    jp::depth_t curDepth = depth->operator()(y, x);
		    if(curDepth > 0) scale = 1.f / curDepth * 1000.f; // depth in mm
		}	      
	      
		size_t leaf = getLeaf([&](const T& test)
		{
		    return test(x, y, scale, data);
		});
		leafs(y, x) = leaf;
	    }
        
        
	}
	
	/**
	 * @brief Store the tree in the file.
	 * 
	 * @param file File stream to store the tree in.
	 * 
	 * @return void
	 */
	void store(std::ofstream& file) const
	{
	    jp::write(file, splits);
	}

	/**
	 * @brief Read the tree from the given file.
	 * 
	 * @param file File stream in which the tree is stored in.
	 * 
	 * @return void
	 */	
	void restore(std::ifstream& file)
	{
	    jp::read(file, splits);
	}
	
	/**
	 * @brief Prints the tree (its split nodes) to the console.
	 * 
	 * @return void
	 */
	void print() const
	{
	    for(unsigned i = 0; i < splits.size(); i++)
	    {
		std::cout << "Node: " << std::endl;
		std::cout << "Left child: " << splits[i].left << std::endl;
		std::cout << "Right child: " << splits[i].right << std::endl;
		std::cout << "Feature: " << std::endl; 
		splits[i].data.print();
		std::cout << std::endl;
	    }
	}

	std::vector<split_t<T>> splits; // list of split nodes (inner nodes)
    };

    /**
     * @brief Write a split node to a file.
     * 
     * @param file File stream to store the split node in.
     * @param split Split node to store.
     * 
     * @return void
     */    
    template <class TFeature>
    void write(std::ofstream& file, const typename jp::split_t<TFeature>& split)
    {
	write(file, split.left);
	write(file, split.right);
	write(file, split.feature);
    }

    /**
     * @brief Read a split node from a file.
     * 
     * @param file File stream to read from.
     * @param split Output parameter. Split node to read.
     * 
     * @return void
     */    
    template <class TFeature>
    void read(std::ifstream& file, typename jp::split_t<TFeature>& split)
    {
	read(file, split.left);
	read(file, split.right);
	read(file, split.feature);      
    }

    /**
     * @brief Class that trains a decision tree structure from data.
     * 
     */    
    template <typename TFeatureSampler>
    class TreeTrainer
    {
    public:
	typedef typename TFeatureSampler::feature_t feature_t; // feature type to use 
	typedef jp::TreeStructure<feature_t> structure_t; // binary decision tree structure type
		
	/**
	 * @brief Type that combines different statistics of leafs during training.
	 */
	struct leaf_stat_t
	{
	    leaf_stat_t() : terminated(false), splitIdx(-1), parent(-1), direction(false) {}
	    leaf_stat_t(int parent, bool direction) : terminated(false), splitIdx(-1), parent(parent), direction(direction) {}

	    int parent; // parent node
	    bool direction; // direction from parent node (right = true)
	    
	    int splitIdx; // mapping to list of splits to process
	    bool terminated; // do not split further
	};
	
	/**
	 * @brief Type that combines different statistics of nodes to be split during training.
	 */	
	struct split_stat_t
	{
	    split_stat_t() : leafIdx(-1) {}
	    split_stat_t(int leafIdx) : leafIdx(leafIdx) {}
	  
	    int leafIdx; // mapping to list of all leafs 
	    std::vector<std::pair<unsigned, sample_t>> leafPixels; // list of pixels that arrived at this leaf
	    
	    feature_t feature; // selected feature
	    double score; // score achieved by selected feature
	};
	
	/**
	 * @brief Constructor.
	 * 
	 * @param imgCache Pointer to the data and ground truth container.
	 * @param sampler Generator of random features.
	 */
	TreeTrainer(const jp::ImageCache* imgCache, const TFeatureSampler& sampler) : 
	    imgCache(imgCache), sampler(sampler)
	{   
	    gp = GlobalProperties::getInstance();
	}

	
	/**
	 * @brief Trains one more depth level of the tree.
	 * 
	 * @param structure Current structure of the decision tree.
	 * @param leafStats Leaf statistics of the current state of the decision tree.
	 * 
	 * @return bool False in case stopping criteria have been met.
	 */
	bool trainingRound(structure_t& structure, std::vector<leaf_stat_t>& leafStats)
	{
	    // collect all leafs that could be split further
	    std::vector<split_stat_t> splitCandidates;

	    for(int leafIdx = 0; leafIdx < leafStats.size(); leafIdx++)
	    {
		// check termination criteria
		if(!leafStats[leafIdx].terminated)
		{
		    leafStats[leafIdx].splitIdx = splitCandidates.size();
		    splitCandidates.push_back(split_stat_t(leafIdx));
		}
		else
		{
		    leafStats[leafIdx].splitIdx = -1;
		}
	    }
	    
	    if(splitCandidates.empty()) return false; //nothing left to split, done
	    
	    // push samples through tree and collect them at the leafs (image ID and pixel position)
	    std::cout << std::endl << "Evaluating tree..." << std::endl;

	    //iteration through images
	    for(unsigned i = 0; i < imgCache->dataCache.size(); i++)
	    {
		std::vector<sample_t> sampling(imgCache->sampleCounts[i]);
		const jp::img_depth_t* imgDepth = NULL;
		if(gp->fP.useDepth) imgDepth = &(imgCache->depthCache[i]);
		
		// half bg hard negative mining
		if(i < imgCache->bgPointer)
		    samplePixels(imgCache->sampleSegCache[i], imgCache->poseCache[i], sampling, imgDepth);
		else
		{
		    // sample 50% of patches according to probability of last auto-context layer
		    // sample other 50% uniformly
		    std::vector<sample_t> sampling1(imgCache->sampleCounts[i] / 2);
		    samplePixels(imgCache->sampleSegCache[i], imgCache->poseCache[i], sampling1, imgDepth);
		    std::vector<sample_t> sampling2(imgCache->sampleCounts[i] / 2);
		    samplePixels(imgCache->objProbCache[i], imgCache->poseCache[i], sampling2, imgDepth);
		    sampling.clear();
		    sampling.insert(sampling.end(), sampling1.begin(), sampling1.end());
		    sampling.insert(sampling.end(), sampling2.begin(), sampling2.end());
		}
		
		//iterate through sampled pixels of image
		for(auto var = sampling.begin(); var != sampling.end(); ++var)
		{
		    // get pixel position
		    sample_t px = *var;
		  
		    // evaluate (current state of) tree for pixel
		    size_t leafIndex = structure.getLeaf([&](const feature_t& feature) 
		    { 
			return feature(px.x, px.y, px.scale, imgCache->dataCache[i]); 
		    });

		    // map index of leaf to index of split candidate, -1 if not a split candidate
		    int splitIdx = leafStats[leafIndex].splitIdx;
    
		    if(splitIdx >= 0) // checks whether this is a split candidate
			splitCandidates[splitIdx].leafPixels.push_back(std::pair<unsigned, sample_t>(i, px));
		}
	    }
	    
	    // first find the best splits for each node, then in a second pass grow the tree
	    std::cout << "Processing " << splitCandidates.size() << " leafs:" << std::endl;
	  
	    std::vector<feature_t> features = sampleFromRandomPixels(
		imgCache->dataCache, 
		gp->fP.featureCount, 
		sampler);

	    for(unsigned splitIdx = 0; splitIdx < splitCandidates.size(); splitIdx++)
	    {
		std::cout << ".";
		std::cout.flush();

		// calculate scores for all features
		std::vector<float> featureScores(gp->fP.featureCount, 0.f);

		#pragma omp parallel for
		for(unsigned featureIndex = 0; featureIndex < features.size(); featureIndex++)
		{	
		    histogram_t histLeft(gp->fP.getLabelCount(), 0);
		    histogram_t histRight(gp->fP.getLabelCount(), 0);
		  
		    for(unsigned pxIdx = 0; pxIdx < splitCandidates[splitIdx].leafPixels.size(); pxIdx++)
		    {
			unsigned imgID = splitCandidates[splitIdx].leafPixels[pxIdx].first;
			sample_t pixelPos = splitCandidates[splitIdx].leafPixels[pxIdx].second;

			// in case of background set label 0
			jp::label_t label = (imgID >= imgCache->bgPointer) ? 0 : imgCache->gtCache[imgID](pixelPos.y, pixelPos.x);
			bool response = features[featureIndex](pixelPos.x, pixelPos.y, pixelPos.scale, imgCache->dataCache[imgID]);
			
			if(response)
			    histRight[label]++;
			else
			    histLeft[label]++;
		    }

		    double score = 0.0;
		    if(histogramTotal(histLeft) > gp->fP.minSamples && histogramTotal(histRight) > gp->fP.minSamples)
			score = informationGain(histLeft, histRight);
		    
		    featureScores[featureIndex] = score;
		}

		// select best feature
		double bestScore = 0.0;
		unsigned bestFeature = 0;		
		
		for(unsigned s = 0; s < featureScores.size(); s++)
		{
		    if(featureScores[s] > bestScore)
		    {
			bestScore = featureScores[s];
			bestFeature = s;
		    }		  
		}
		
		splitCandidates[splitIdx].feature = features[bestFeature];
		splitCandidates[splitIdx].score = bestScore;
	    }
	    
	    //grow the tree
	    std::cout << std::endl << "Splitting leaf nodes ..." << std::endl;
	    double minScore = 0.0;
	    bool newLayer = false;
	    
	    for(unsigned splitIdx = 0; splitIdx < splitCandidates.size(); splitIdx++)
	    {
		int leafIdx = splitCandidates[splitIdx].leafIdx;
		int parent = leafStats[leafIdx].parent;
		bool direction = leafStats[leafIdx].direction;
		
		if(splitCandidates[splitIdx].score > minScore)
		{
		    size_t leafLeft = structure.empty()
			? 0 
			: structure.child2Leaf(structure.getSplitChild(parent, direction));
		    size_t leafRight = leafStats.size();
		      
		    // add new node
		    int split = structure.addNode(splitCandidates[splitIdx].feature);
		    structure.setSplitChild(split, false, structure.leaf2Child(leafLeft));
		    structure.setSplitChild(split, true, structure.leaf2Child(leafRight));		    
		    	    
		    // link new node to parent (unless it is root node)
		    if(structure.size() - 1 > 0) structure.setSplitChild(parent, direction, split);
		    
		    // init new leaves
		    leafStats[leafLeft] = leaf_stat_t(split, false);
		    leafStats.push_back(leaf_stat_t(split, true));
		    		    
		    featuresChosen[splitCandidates[splitIdx].feature.getType()]++;
		    if(!newLayer)
		    {
			newLayer = true;
			treeDepth++;
		    }
		}
		else
		{
		    size_t leafIndex = structure.empty()
			? 0 
			: structure.child2Leaf(structure.getSplitChild(parent, direction));
		    leafStats[leafIndex].terminated = true;
		}
	    }
	    std::cout << std::endl;
	    
	    std::cout << "Tree depth: " << treeDepth << std::endl;
	    return (treeDepth < gp->fP.maxDepth); // continue if max depth has not been reached
	}

	
	/**
	 * @brief Train a decision tree structure from data.
	 * 
	 * @param structure Output parameter. Structure to be trained.
	 * @return void
	 */
	void train(jp::TreeStructure<feature_t>& structure)
	{
	    // clear training info (from last train call)
	    featuresChosen.clear();
	    treeDepth = 0;
	    
	    // initialize tree structure and training info   
	    structure.splits.clear();
	    std::vector<leaf_stat_t> leafStats(1, leaf_stat_t());
	    
	    // train tree breath first
	    while(true)
	    {
		if(!trainingRound(structure, leafStats))
		    break;
		std::cout << "Tree now has " << structure.getNodeCount() << " nodes." << std::endl;
	    }

	    std::cout << std::endl << "------------------------------------------------" << std::endl;
	    
	    std::cout << std::endl << "Features chosen: " << std::endl;
	    for(auto pair : featuresChosen)
	    {
		if(pair.first == 0) std::cout << "Color: " << pair.second << std::endl;
		else if(pair.first == 9) std::cout << "Abs Cell: " << pair.second << std::endl;
		else if(pair.first == 10) std::cout << "Abs Cell: " << pair.second << std::endl;
		else std::cout << "Unknown: " << pair.second << std::endl;
	    }
	}

    private:

      /**
       * @brief Returns the sum of histogram entries.
       * 
       * @param h Input Histogram.
       */
      inline unsigned long histogramTotal(const histogram_t& h) const
	{
	  long total = 0;
	  for(label_t i = 0; i < h.size(); i++)
	      total += h[i];
	  return total;
	}
      
      /**
       * @brief Returns the Shannon entropy of the histogram.
       * 
       * @param h Input Histogram.
       */
      inline double histogramEntropy(const histogram_t& h) const
	{
	    double entropy = 0.0;
	    double scale = 1.0 / (histogramTotal(h) + EPS * h.size());
	    for(label_t label = 0; label < h.size(); label++)
	    {
		double f = scale * (EPS + h[label]);
		entropy -= f * std::log(f);
	    }
	    return entropy;
	}
	
      /**
       * @brief Returns the information gain given a histogram split.
       * 
       * The parent histogram is calculated from the two children.
       * 
       * @param left Child histogram 1
       * @param right Child histogram 2
       */	
	double informationGain(const histogram_t& left, const histogram_t& right) const
	{
	    assert(left.size() == right.size());
	    unsigned long totalLeft = histogramTotal(left);
	    unsigned long totalRight = histogramTotal(right);
	    unsigned long total = totalLeft + totalRight;
	    if(totalLeft == 0 || totalRight == 0) return 0.0;

	    histogram_t parent(left.size(), 0);
	    for(label_t label = 0; label < parent.size(); label++)
		parent[label] = left[label] + right[label];

	    double inv = 1.0 / total;
	    return histogramEntropy(parent) - inv * 
		(histogramEntropy(left) * totalLeft + histogramEntropy(right) * totalRight);
	}
      
	const jp::ImageCache* imgCache; // data container (including ground truth)
	const TFeatureSampler sampler; // random feature generator
	const GlobalProperties* gp; // pointer to global configuration parameters
	
	std::map<int, int> featuresChosen; // feature type: how often it was selected
	unsigned treeDepth; // current tree depth
    };    
}

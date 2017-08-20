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
#include "features.h"

/** Generic feature class and sampler class that hide specific feature 
 * types. Used to combine different features into one type. This file 
 * also includes the definition of conrete feature type combination used 
 * throughout the code.*/

namespace jp
{
    /**
     * @brief Generic feature class that is a transparent interface (or wrapper) to two specific feature sub classes. 
     * 
     * Can be used to combine differnt features into one type. An object 
     * of this class can be an object of either of the two template sub 
     * feature types. For more than 2 specific sub feature types a recursive binary 
     * tree of feature types can be constructed.
     */
    template<typename TFeature1, typename TFeature2>
    class FeatureCombined
    {
    public:
        /**
         * @brief Default constructor.
         */
        FeatureCombined() {}
      
        /**
         * @brief Constructor to make this feature hide a feature of the first type.
         * 
	 * @param feat1 Feature of type 2.
         */
	FeatureCombined(const TFeature1& feat1) : feat1(feat1), selectFirst(true)
	{
	}
	
        /**
         * @brief Constructor to make this feature hide a feature of the second type.
         * 
	 * @param feat2 Feature of type 2.
         */	
	FeatureCombined(const TFeature2& feat2) : feat2(feat2), selectFirst(false)
	{
	}

	/**
	 * @brief Returns a feature type ID. This will return the type ID of the specific feature that this wrapper hides.
	 * 
	 * @return uchar Feature type ID.
	 */
	unsigned char getType() const
	{
	    return (selectFirst) ? feat1.getType() : feat2.getType();
	}
	
	/**
	 * @brief Set the threshold of the feature hidden.
	 * 
	 * @ thresh Threshold.
	 * @return void
	 */
	void setThreshold(double thresh)
	{
	    (selectFirst) ? feat1.setThreshold(thresh) : feat2.setThreshold(thresh);
	}
	
	/**
	 * @brief Compute the response of the feature that this wrapper hides.
	 * 
	 * @param x X component of the center pixel.
	 * @param y Y component of the center pixel.
	 * @param scale Feature offsets are scaled by this factor.
	 * @param data Input frame.
	 * @return double Feature response.
	 */
	double computeResponse(int x, int y, float scale, const jp::img_data_t& data) const
	{   
	    return (selectFirst) ? 
		feat1.computeResponse(x, y, scale, data): 
		feat2.computeResponse(x, y, scale, data);     
	}
	
	/**
	 * @brief Calculates the feature response of the feature this wraper hides and compares it to the feature threshold.
	 * 
	 * @param x X component of the center pixel.
	 * @param y Y component of the center pixel.
	 * @param scale Feature offsets are scaled by this factor.
	 * @param data Input frame.
	 * @return bool False if feature response if below threshold.
	 */	
	bool operator()(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    return (selectFirst) ?
		feat1(x, y, scale, data):
		feat2(x, y, scale, data);
	}
	
	/**
	 * @brief Write the wrapper feature and the hidden feature to the given file.
	 * 
	 * @param file File to write to.
	 * @return void
	 */	
	void store(std::ofstream& file) const
	{
	    write(file, selectFirst);
	    
	    (selectFirst) ? 
		write(file, feat1):
		write(file, feat2);
	}

	/**
	 * @brief Read the wrapper feature and the hidden feature from the given file.
	 * 
	 * @param file File to read from.
	 * @return void
	 */	
	void restore(std::ifstream& file)
	{
	    read(file, selectFirst);
	  
	    (selectFirst) ? 
		read(file, feat1):
		read(file, feat2);
	}
	
	/**
	 * @brief Print the parameters of the feature this wrapper hides to the console.
	 * 
	 * @return void
	 */	
	void print() const
	{
	    (selectFirst) ? 
		feat1.print(): 
		feat2.print();
	}
      
    private:
	TFeature1 feat1; // feature of type 1, either feat1 or feat2 is set
	TFeature2 feat2; // feature of type 2, either feat1 or feat2 is set
	bool selectFirst; // marks whether feat1 or feat2 is set
    };

    // Corresponding feature sampler, initialized with the two feature samplers and
    // the fraction of feature1 types sampled.
    
    /**
     * @brief Sampler class of the wrapper feature. 
     * 
     * Used to create random features. The sampler will decide randomly which sub-type a wrapper feature hides.
     */    
    template<typename TFeatureSampler1, typename TFeatureSampler2>
    class FeatureSamplerCombined
    {
    public:
	typedef typename TFeatureSampler1::feature_t feature_t1;
	typedef typename TFeatureSampler2::feature_t feature_t2;
	typedef FeatureCombined<feature_t1, feature_t2> feature_t;

	/**
	 * @brief Constructor.
	 * 
	 * @sampler1 Sampler of feature sub type 1.
	 * @sampler2 Sampler of feature sub type 2.
	 * @fracFirst Probability to generate a feature of type 1.
	 */
	FeatureSamplerCombined(const TFeatureSampler1& sampler1, const TFeatureSampler2& sampler2,
	    double fracFirst = 0.5)
	    : sampler1(sampler1), sampler2(sampler2), fracFirst(fracFirst)
	{
	    assert(fracFirst >= 0.0 && fracFirst <= 1.0);
	}

	/**
	 * @brief Generate a random feature. 
	 * 
	 * First its randomly decided which sub type to sample, and then a random feature of this type is sampled.
	 * 
	 * @return feature_t Feature.
	 */
	feature_t sampleFeature() const
	{
	    if (drand(0, 1) <= fracFirst)
		return feature_t(sampler1.sampleFeature());
	    else
		return feature_t(sampler2.sampleFeature());
	}
	
	/**
	 * @brief Generate the same random feature multiple times.
	 * 
	 * Feature type and feature parameters are sampled once and multiple features are generated according to these parameters.
	 * 
	 * @return std::vector< feature_t > List of identical features.
	 */	
	std::vector<feature_t> sampleFeatures(unsigned count) const
	{
	    std::vector<feature_t> features;
	    features.reserve(count);
	  
	    if (drand(0, 1) <= fracFirst)
	    {
		std::vector<feature_t1> features1 = sampler1.sampleFeatures(count);
		for(unsigned i = 0; i < features1.size(); i++)
		    features.push_back(feature_t(features1[i]));
	    }
	    else
	    {
		std::vector<feature_t2> features2 = sampler2.sampleFeatures(count);
		for(unsigned i = 0; i < features2.size(); i++)
		    features.push_back(feature_t(features2[i]));
	    }
	    
	    return features;
	}   
      
    private:
	std::uniform_real_distribution<double> rfirst; // distribution of sampling a feature of the first or second type

	TFeatureSampler1 sampler1; // sampler of feature type 1
	TFeatureSampler2 sampler2; // sampler of feature type 2
	double fracFirst; // probability of sampling feature type 1
    };

    /**
     * @brief Writing a complex feature to the given file.
     * 
     * @param file File to write to.
     * @param feature Feature to write.
     * @return void
     */
    template<typename TFeature1, typename TFeature2>
    void write(std::ofstream& file, const FeatureCombined<TFeature1, TFeature2>& feature)
    {
	feature.store(file);
    }
    
    /**
     * @brief Reading a complex feature from the given file.
     * 
     * @param file File to read from.
     * @param feature Feature to read.
     * @return void
     */    
    template<typename TFeature1, typename TFeature2>
    void read(std::ifstream& file, FeatureCombined<TFeature1, TFeature2>& feature)
    {
	feature.restore(file);
    }
    
    typedef FeatureSamplerCombined<FeatureSamplerAbsCoord, FeatureSamplerAbsCell> sampler_inner_t1; // inner node of recursive feature type tree
    typedef FeatureSamplerCombined<FeatureSamplerDABGR, sampler_inner_t1> sampler_outer_t; // inner node of recursive feature type tree
    typedef sampler_outer_t::feature_t feature_t; // feature type definition used throughout this code
}
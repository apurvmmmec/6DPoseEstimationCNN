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

/** Auto context object class feature and feature sampler */

namespace jp
{
    /**
     * @brief Random forest feature that probes an object class on the auto-context feature channels.
     */    
    class FeatureAbsCell
    {
    public:
      
        /**
         * @brief Default contructor.
         */            
	FeatureAbsCell() : off_x(0), off_y(0), channel(0), direction(0)
	{
	}

	/**
	 * @brief Constructs a feature.
	 * 
	 * @param off_x X component of the offset vector.
	 * @param off_y Y component of the offset vector.
	 * @param channel Which auto context feature channel to probe (channels correspond to different objects).
	 * @param direction Should the feature response be smaller or greater than the threshold?
	 */	
	FeatureAbsCell(int off_x, int off_y, int channel, int direction) 
	    : off_x(off_x), off_y(off_y), channel(channel), direction(direction)
	{
	}

	/**
	 * @brief Returns a feature type ID.
	 * 
	 * @return uchar Feature type ID.
	 */		
	uchar getType() const { return 9; }

	/**
	 * @brief Computes the feature value for the given center pixel. Feature offsets are scaled by the given scale factor.
	 * 
	 * @param x X component of the center pixel.
	 * @param y Y component of the center pixel.
	 * @param scale Feature offsets are scaled by this factor.
	 * @param data Input frame.
	 * @return double Feature response.
	 */		
	double computeResponse(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    // auto-context feature channels might be stored sub-sampled, in this case the offset vector has also be sub-sampled
	    int acSubSample = GlobalProperties::getInstance()->fP.acSubsample;
	    
	    // scale and clamp the offset vector
	    FeaturePoints fP = getFeaturePoints(
		x/acSubSample, 
		y/acSubSample, 
		off_x/acSubSample, 
		off_y/acSubSample, 
		scale/acSubSample, 
		data.labelData[channel].cols, 
		data.labelData[channel].rows);
	    
	    // probe the auto-context object class feature channel
	    return data.labelData[channel](fP.y1, fP.x1);
	}

	/**
	 * @brief Calculates the feature response and compares it to the feature threshold.
	 * 
	 * @param x X component of the center pixel.
	 * @param y Y component of the center pixel.
	 * @param scale Feature offsets are scaled by this factor.
	 * @param data Input frame.
	 * @return bool False if feature response if below threshold. Behaviour can be flipped if direction is 0.
	 */	
	bool operator()(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    if(direction)
		return computeResponse(x, y, scale, data) <= thresh;
	    else
		return computeResponse(x, y, scale, data) > thresh;
	}

	/**
	 * @brief Set the feature threshold.
	 * 
	 * @return void
	 */		
	void setThreshold(double thresh) { this->thresh = thresh; }

	/**
	 * @brief Print the parameters of the feature to the console.
	 * 
	 * @return void
	 */	
	void print() const
	{
	    std::cout << "Absolute Cell Feature (x: " << off_x << ", y: " << off_y 
		<< ", t: " << thresh << ")" << std::endl;
	}
	
	/**
	 * @brief Write the feature to the given file.
	 * 
	 * @param file File to write to.
	 * @return void
	 */	
	void store(std::ofstream& file) const
	{
	    write(file, off_x);
	    write(file, off_y);
	    write(file, thresh);
	    write(file, channel);
	    write(file, direction);
	}

	/**
	 * @brief Read the feature from the given file.
	 * 
	 * @param file File to read from.
	 * @return void
	 */		
	void restore(std::ifstream& file)
	{
	    read(file, off_x);
	    read(file, off_y);
	    read(file, thresh);
	    read(file, channel);
	    read(file, direction);
	}
	
    private:
	int off_x, off_y;  // offset vector of the pixel probe
	int channel; // which auto context feature channel to probe (channels correspond to different objects)
	jp::label_t thresh; // feature threshold
	int direction; //should the feature response be smaller or greater than the threshold?
    };

    /**
     * @brief Write a feature to the given file.
     * 
     * @param file File to write to.
     * @param feature Feature to write.
     * @return void
     */        
    template<>
    void write(std::ofstream& file, const FeatureAbsCell& feature);
  
    /**
     * @brief Read a feature from the given file.
     * 
     * @param file File to read from.
     * @param feature Feature read from.
     * @return void
     */    
    template<>
    void read(std::ifstream& file, FeatureAbsCell& feature);
    
    /**
     * @brief Class to randomly sample auto-context object class features.
     */        
    class FeatureSamplerAbsCell
    {
    public:
	typedef FeatureAbsCell feature_t;

	/**
	 * @brief Constructor. The (x,y) offsets for the feature tests are sampled from a uniform distribution
	 * from -off_max to off_max. 
	 * 
	 * @param off_max Maximally allowed feature offset.
	 * @param maxChannel Number of auto-context feature channels. One will be randomly chosen.
	 */
	FeatureSamplerAbsCell(int off_max, int maxChannel) : off_max(off_max), maxChannel(maxChannel)
	{
	}
	    
	/**
	  * @brief Generate one random feature.
	  * 
	  * @return jp::FeatureSamplerAbsCell::feature_t Random feature.
	  */	 	    
	feature_t sampleFeature() const
	{
	    return feature_t(getOffset(), getOffset(), getChannel(), getDirection());
	}

	/**
	 * @brief Create a number of IDENTICAL feautures.
	 * 
	 * Feature parameters will be choosen randomly once and set for all features.
	 * 
	 * @param count How meany features to generate?
	 * @return std::vector< feature_t > List of identical features.
	 */	
	std::vector<feature_t> sampleFeatures(unsigned count) const
	{
	    // create number of thresholds of identical features
	    int offset1 = getOffset();
	    int offset2 = getOffset();
	    int channel = getChannel();
	    int direction = getDirection();

	    std::vector<feature_t> features;
	    for(unsigned i = 0; i < count; i++)
	    {
		features.push_back(feature_t(offset1, offset2, channel, direction));
	    }
		
	    return features;
	}

    private:
	int off_max;  // maximally allowed offset vector
	int maxChannel; // number of auto-context feature channels
	
	/**
	 * @brief Returns a random pixel offset.
	 * 
	 * @return int Offset.
	 */	
	int getOffset() const { return irand(-off_max, off_max + 1); }
	
	/**
	 * @brief Returns a random auto-context feature channel number.
	 * 
	 * @return int Channel number.
	 */		
	int getChannel() const { return irand(0, maxChannel); }
	
	/**
	 * @brief Returns a random direction (0,1) of how feature thresholds should be compared.
	 * 
	 * @return int Direction.
	 */	
	int getDirection() const { return irand(0, 2); }
    };  
}
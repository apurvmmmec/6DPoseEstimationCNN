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

#include "read_data.h"
#include "util.h"

#include <fstream>
#include "png++/png.hpp"

namespace jp
{
    void readData(const std::string dFile, jp::img_depth_t& image)
    {
	png::image<depth_t> imgPng(dFile);
	image = jp::img_depth_t(imgPng.get_height(), imgPng.get_width());

	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	{
	    image(y, x) = (jp::depth_t) imgPng.get_pixel(x, y);
	}
    }
    
    void readData(const std::string labelFile, jp::img_cordL_t& image)
    {
        png::image<gray_t> imgPng(labelFile);
        image = jp::img_cordL_t(imgPng.get_height(), imgPng.get_width());
        
        for(int x = 0; x < imgPng.get_width(); x++)
            for(int y = 0; y < imgPng.get_height(); y++)
            {
                image(y, x) = (jp::gray_t) imgPng.get_pixel(x, y);
            }
    }
    
    void readData(const std::string bgrFile, jp::img_bgr_t& image)
    {
	png::image<png::basic_rgb_pixel<uchar>> imgPng(bgrFile);
	image = jp::img_bgr_t(imgPng.get_height(), imgPng.get_width());
	
	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	{
	    image(y, x)(0) = (uchar) imgPng.get_pixel(x, y).blue;
	    image(y, x)(1) = (uchar) imgPng.get_pixel(x, y).green;
	    image(y, x)(2) = (uchar) imgPng.get_pixel(x, y).red;
	}
    }
  
    void readData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image)
    {
	readData(bgrFile, image.bgr);
	readData(dFile, image.depth);
    }

    void readData(const std::string coordFile, jp::img_coord_t& image)
    {
	png::image<png::basic_rgb_pixel<unsigned short>> imgPng(coordFile);
	image = jp::img_coord_t(imgPng.get_height(), imgPng.get_width());
	
	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	{
//        std::cout<<imgPng.get_pixel(x, y).red<<std::endl;
//        std::cout<<imgPng.get_pixel(x, y).green<<std::endl;
//        std::cout<<imgPng.get_pixel(x, y).blue<<std::endl;

	    image(y, x)(0) = (jp::coord1_t) imgPng.get_pixel(x, y).red;
	    image(y, x)(1) = (jp::coord1_t) imgPng.get_pixel(x, y).green;
	    image(y, x)(2) = (jp::coord1_t) imgPng.get_pixel(x, y).blue;
//        std::cout<<image(y, x)(0)<<std::endl;
//        std::cout<<image(y, x)(0)<<std::endl;
//        std::cout<<image(y, x)(0)<<std::endl;
	}
    }
    
   
    bool readData(const std::string infoFile, jp::info_t& info)
    {
        std::cout<<infoFile<<std::endl;
	std::ifstream file(infoFile);
	if(!file.is_open())
	{
	    info.visible = false;
	    return false;
	}
	
	std::string line;
	int lineCount = 0;
	std::vector<std::string> tokens;
	
	while(true)
	{
	    std::getline(file, line);
	    tokens = split(line);
	    
	    if(file.eof())	
	    {
		info.visible = false;
		return false;
	    }
	    if(tokens.empty()) continue;
	    lineCount++;
	    
	    if(lineCount == 3) info.name = tokens[0];
	    
	    if(tokens[0] == "occlusion:")
	    {
		std::getline(file, line);
		info.occlusion = (float) atof(line.c_str());
	    }	
	    else if(tokens[0] == "rotation:")
	    {
		info.rotation = cv::Mat_<float>(3, 3);
	      
		for(unsigned i = 0; i < 3; i++)
		{
		    std::getline(file, line);
		    tokens = split(line);
		    
		    for(unsigned j = 0; j < 3; j++) 
			info.rotation(i, j) = (float) atof(tokens[j].c_str());
		}
	    }
	    else if(tokens[0] == "center:")
	    {
		std::getline(file, line);
		tokens = split(line);	      
	      
		for(unsigned j = 0; j < 3; j++) 
		    info.center(j) = (float) atof(tokens[j].c_str());
	    }
	    else if(tokens[0] == "extent:" || tokens[0] == "extend:") // there was a typo in some of our files ;)
	    {
		std::getline(file, line);
		tokens = split(line);	      
	      
		for(unsigned j = 0; j < 3; j++) 
		    info.extent(j) = (float) atof(tokens[j].c_str());
		
		info.visible = true;
		return true;
	    }
    
	}
    }
}



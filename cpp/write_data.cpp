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
    void writeData(const std::string dFile, jp::img_depth_t& image)
    {
	png::image<depth_t> imgPng(image.cols, image.rows);

	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	    imgPng.set_pixel(x, y, image(y, x));

	imgPng.write(dFile);
    }

    void writeData(const std::string bgrFile, jp::img_bgr_t& image)
    {
	cv::imwrite(bgrFile, image);
    }
  
    void writeData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image)
    {
	writeData(bgrFile, image.bgr);
	writeData(dFile, image.depth);
    }

    void writeData(const std::string coordFile, jp::img_coord_t& image)
    {
	png::image<png::basic_rgb_pixel<unsigned short>> imgPng(image.cols, image.rows);
	
	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	{
	    png::basic_rgb_pixel<unsigned short> px;
	    px.red = (unsigned short) image(y, x)(0);
	    px.green = (unsigned short) image(y, x)(1);
	    px.blue = (unsigned short) image(y, x)(2);
	  
	    imgPng.set_pixel(x, y, px);
	}
	
	imgPng.write(coordFile);
    }
    
    void writeData(const std::string infoFile, jp::info_t& info)
    {
	std::ofstream file(infoFile);
	
	file << "image size" << std::endl;
	file << 640 << " " << 480 << std::endl;
	
	if(!info.visible) // info file is left empty in case the object is not visible
	{
	    file.close();
	    return;
	}
	
	file << info.name << std::endl;
	if(info.occlusion != 0) // occlusion information is optional
	{
	    file << "occlusion:" << std::endl;
	    file << info.occlusion << std::endl;
	}
	file << "rotation:" << std::endl;
	file << info.rotation(0, 0) << " " << info.rotation(0, 1) << " " << info.rotation(0, 2) << std::endl;
	file << info.rotation(1, 0) << " " << info.rotation(1, 1) << " " << info.rotation(1, 2) << std::endl;
	file << info.rotation(2, 0) << " " << info.rotation(2, 1) << " " << info.rotation(2, 2) << std::endl;
	file << "center:" << std::endl;
	file << info.center(0) << " " << info.center(1) << " " << info.center(2) << std::endl;
	file << "extent:" << std::endl;
	file << info.extent(0) << " " << info.extent(1) << " " << info.extent(2) << std::endl;
	
	file.close();
    }
}



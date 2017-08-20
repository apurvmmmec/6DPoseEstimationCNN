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
 
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include "Hypothesis.h"
#include "../core/util.h"

/**
 * @brief Reads a point cloud from a file that includes vertex positions (3 values), normals (3 values), and color information (3 values).
 * 
 * The first line in the file is ignored.
 * Color information is ignored.
 * Only each 20th line of the file is read to decrease point cloud size.
 * The point cloud will be scaled by the given scale factor.
 * The point cloud will be translated such that its bounding box is centered at zero.
 * 
 * @param path Point cloud file name including path.
 * @param scale Factor by which to scale the point cloud (e.g. for conversion of meters to millimeters)
 * @param points Output parameter. Point cloud.
 * @param normals Output parameter. Point cloud normals.
 * @return void
 */
void inline loadPointCloud9Cols(std::string path, double scale, std::vector<cv::Point3d>& points, std::vector<cv::Point3d>& normals)
{
    double maxX, maxY, minX, minY, maxZ, minZ;
    std::ifstream myfile(path.c_str());
    
    double x, y, z, nx, ny, nz, r, g, b;
    std::string firstLine;
    
    if (myfile.is_open())
    {
	// remove first line
	std::getline(myfile, firstLine);
	
	bool first = true;
	int i = 0;
	
	while(myfile >> x >> y >> z >> nx >> ny >> nz >> r >> g >> b)
	{
	    if(i%20==0)
	    {
		if(first)
		{
		    maxX = x; minX = x; maxY = y; minY = y; maxZ = z; minZ = z;
		}
	
		first = false;

		//To rotations by 90 degrees; (difference in coordinate system definition)
		std::swap(x, z);
		x *= -1;
		
		std::swap(x, y);
		x *= -1;
		
		std::swap(nx, nz);
		nx *= -1;
		
		std::swap(nx, ny);
		nx *= -1;
		
		maxX = std::max(maxX, x); minX = std::min(minX, x);
		maxY = std::max(maxY, y); minY = std::min(minY, y);
		maxZ = std::max(maxZ, z); minZ = std::min(minZ, z);
		
		points.push_back(cv::Point3d(scale*x,scale*y, scale*z));
		normals.push_back(cv::Point3d(nx, ny, nz));
	    }
	    i++;
	}
	
	myfile.close();
    }
    else std::cout << "Unable to open point cloud file!"; 
    
    cv::Point3d center(scale * (minX + maxX) / 2.0, scale * (minY + maxY) / 2.0, scale * (minZ + maxZ) / 2.0);
    
    for(int i = 0; i < points.size(); i++)
    {
	points[i] = points[i] - center;
    }
}

/**
 * @brief Reads a point cloud from a file of vertex positions.
 * 
 * Only each 10th line of the file is read to decrease point cloud size.
 * The point cloud will be scaled by the given scale factor.
 * The point cloud will be translated such that its bounding box is centered at zero.
 * 
 * @param path Point cloud file name including path.
 * @param scale Optional. Factor by which to scale the point cloud (e.g. for conversion of meters to millimeters)
 * @return std::vector< cv::Point3d, std::allocator< void > > Point cloud.
 */
std::vector<cv::Point3d> inline loadPointCloud(std::string path, double scale = 1)
{
    double maxX, maxY, minX, minY, maxZ, minZ;
    
    std::ifstream myfile(path.c_str());
    std::vector<cv::Point3d> result;
    
    double x,y,z;
    std::string firstLine;
    
    if(myfile.is_open())
    {
	bool first = true;
	int i = 0;
    
	while(myfile >> x >> y >> z)
	{
	    if(i%10==0)
	    {
		if(first)
		{
		    maxX = x; minX = x; maxY = y; minY = y; maxZ = z; minZ = z;
		}
	    
		first = false;

		maxX = std::max(maxX, x); minX = std::min(minX, x);
		maxY = std::max(maxY, y); minY = std::min(minY, y);
		maxZ = std::max(maxZ, z); minZ = std::min(minZ, z);
		
		result.push_back(cv::Point3d(scale * x, scale * y, scale * z));
	    }
	    i++;
	}
	
	myfile.close();
    }
    else std::cout << "Unable to open point cloud file!";  
    
    cv::Point3d center(scale * (minX + maxX) / 2.0, scale * (minY + maxY) / 2.0, scale * (minZ + maxZ) / 2.0);
    
    for(int i = 0; i < result.size(); i++)
    {
	result[i] = result[i] - center;
    }
    return result;
}

/**
 * @brief Reads a point cloud from a file that includes a prefix (1 value), vertex positions (3 values) and color information (3 values).
 * 
 * The prefixes are ignored.
 * Color information is ignored.
 * Only each 10th line of the file is read to decrease point cloud size.
 * The point cloud will be scaled by the given scale factor.
 * The point cloud will be translated such that its bounding box is centered at zero.
 * 
 * @param path Point cloud file name including path.
 * @param scale Optional. Factor by which to scale the point cloud (e.g. for conversion of meters to millimeters)
 * @return std::vector< cv::Point3d, std::allocator< void > > Point cloud.
 */
std::vector<cv::Point3d> inline loadPointCloud6Col(std::string path, double scale = 1)
{
    double maxX, maxY, minX, minY, maxZ, minZ;
    
    std::ifstream myfile(path.c_str());
    std::vector<cv::Point3d> result;
    
    double x, y, z, r, g, b;
    uchar prefix;
    std::string firstLine;
    
    if(myfile.is_open())
    {
	bool first = true;
	int i = 0;
	
	while(myfile >> prefix >> x >> y >> z >> r >> g >> b)
	{
	    if(i % 10 == 0)
	    {
		if(first)
		{
		    maxX = x; minX = x; maxY = y; minY = y; maxZ = z; minZ = z;
		}
		
		first = false;

		maxX = std::max(maxX, x); minX = std::min(minX, x);
		maxY = std::max(maxY, y); minY = std::min(minY, y);
		maxZ = std::max(maxZ, z); minZ = std::min(minZ, z);
		result.push_back(cv::Point3d(scale * x, scale * y, scale * z));
	    }
	    i++;
	}
	  
	myfile.close();
    }
    else std::cout << "Unable to open point cloud file!";  
    
    cv::Point3d center(scale * (minX + maxX) / 2.0, scale * (minY + maxY) / 2.0, scale * (minZ + maxZ) / 2.0);
    
    for(int i = 0; i < result.size(); i++)
    {
	result[i] = result[i] - center;
    }
    return result;
}

/**
 * @brief Loads a point cloud from various file formats that we encountered. The format will be determined by checking the first line.
 * 
 * See the other methods in this file which formats we support. Point cloud will be in millimeters and its bounding box centered at zero.
 * 
 * @param path Point cloud file name including path.
 * @param points Output parameter. Point cloud.
 * @param maxPoints If the point cloud has more points than this, it will be randomly sub sampled.
 * @return void
 */
void inline loadPointCloudGeneric(std::string path, std::vector<cv::Point3d>& points, int maxPoints)
{
  // check first line of file to decide on the file format
  std::ifstream myfile(path.c_str());
  
  if(myfile.is_open())
  {
      std::string firstLine;
      std::getline(myfile, firstLine);
      myfile.close();
      
      std::vector<std::string> tokens = split(firstLine);
	  
      if(tokens.size() == 2) // hinterstoisser file format
      {
	  std::vector<cv::Point3d> normalCloud;
	  loadPointCloud9Cols(path, 10, points, normalCloud);
      }
      else if(tokens.size() == 3) // franks file format
      {
	  points = loadPointCloud(path, 1000);
      }
      else 
      {
	  points = loadPointCloud6Col(path, 1);
      }
  }
  
  if(points.size() < maxPoints) 
      return;
  
  std::vector<cv::Point3d> filteredPoints;
  for(unsigned i = 0; i < maxPoints; i++)
      filteredPoints.push_back(points[irand(0, points.size())]);

  points = filteredPoints;
}
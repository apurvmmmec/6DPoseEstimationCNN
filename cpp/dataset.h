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

#include "properties.h"
#include "util.h"
#include "read_data.h"
#include <stdexcept>

/** Interface for reading and writing datasets and some basis operations.*/

namespace jp
{
    /**
     * @brief Create a global label from the object ID and object specific quantized object coordinate labels.
     * 
     * @param objID Object ID.
     * @param objCell Object specific label.
     * @return jp::label_t Global label.
     */
    jp::label_t getLabel(jp::id_t objID, jp::cell_t objCell);

    /**
     * @brief Calculate the camera coordinate given a pixel position and a depth value.
     * 
     * @param x X component of the pixel position.
     * @param y Y component of the pixel position.
     * @param depth Depth value at that position in mm.
     * @return jp::coord3_t Camera coordinate.
     */
    jp::coord3_t pxToEye(int x, int y, jp::depth_t depth);

    /**
     * @brief Checks whether the given object coordinate lies on the object (is not 0 0 0).
     * 
     * @param pt Object coordinate.
     * @return bool True if not background.
     */
    bool onObj(const jp::coord3_t& pt); 
    
    /**
     * @brief Checks whether the given label on the object (is not background).
     * 
     * @param pt Label.
     * @return bool True if not background.
     */    
    bool onObj(const jp::label_t& pt);
    
    /**
     * @brief Class that is a interface for reading and writing object specific data.
     * 
     */
    class Dataset
    {
    public:

      	Dataset()
	{
	}
      
	/**
	 * @brief Constructor.
	 * 
	 * @param basePath The directory where there are subdirectories "rgb_noseg", "depth_noseg", "seg", "obj", "info".
	 * @param objID Object ID this dataset belongs to.
	 */
	Dataset(const std::string& basePath, jp::label_t objID) : objID(objID)
	{
	    readFileNames(basePath);
	}

	/**
	 * @brief Return the object ID this dataset belongs to.
	 * 
	 * @return jp::id_t Object ID.
	 */
	jp::id_t getObjID() const
	{
	    return objID;
	}
	
	/**
	 * @brief Size of the dataset (number of frames).
	 * 
	 * @return size_t Size.
	 */
	size_t size() const 
	{ 
	    return bgrFiles.size();
	}

	/**
	 * @brief Return the RGB image file name of the given frame number.
	 * 
	 * @param i Frame number.
	 * @return std::string File name.
	 */
	std::string getFileName(size_t i) const
	{
	    return bgrFiles[i];
	}
	
	/**
	 * @brief Get ground truth information for the given frame.
	 * 
	 * @param i Frame number.
	 * @return bool Returns if there is no valid ground truth for this frame (object not visible).
	 */
	bool getInfo(size_t i, jp::info_t& info) const
	{
	    if(infoFiles.empty()) return false;
	    if(!readData(infoFiles[i], info))
		return false;
	    return true;
	}	
	
	/**
	 * @brief Get the RGB image of the given frame.
	 * 
	 * If RGB and depth is not registered (according to GlobalProperties) RGB will be coarsly aligned with depth (rescaled and shifted).
	 * 
	 * @param i Frame number.
	 * @param img Output parameter. RGB image.
	 * @param noseg If true, image will not be segmented.
	 * @return void
	 */
	void getBGR(size_t i, jp::img_bgr_t& img, bool noseg) const
	{
	    std::string bgrFile = bgrFiles[i];
	    
	    readData(bgrFile, img);
	    
	    if(!noseg) // segment the image according to ground truth segmentation
	    {
		jp::img_id_t seg;
		getSegmentation(i, seg);
		
		for(unsigned x = 0; x < img.cols; x++)
		for(unsigned y = 0; y < img.rows; y++)
		    if(!seg(y, x)) img(y, x) = jp::bgr_t(0, 0, 0);
	    }
	    
	    GlobalProperties* gp = GlobalProperties::getInstance();
	    if(!gp->fP.rawData) return;
	    
	    // coarsly register RGB and depth
	    float scaleFactor = gp->fP.focalLength / gp->fP.secondaryFocalLength; // rescale RGB image according to the difference of focal lengths of the different sensors
	    float transCorrX = (1 - scaleFactor) * (gp->fP.imageWidth * 0.5 + gp->fP.rawXShift); // shift RGB channel based on manual parameters
	    float transCorrY = (1 - scaleFactor) * (gp->fP.imageHeight * 0.5 + gp->fP.rawYShift); // shift RGB channel based on manual parameters
		
	    // apply transformation
	    cv::Mat trans = cv::Mat::eye(2, 3, CV_64F) * scaleFactor;
	    trans.at<double>(0, 2) = transCorrX;
	    trans.at<double>(1, 2) = transCorrY;
	    jp::img_bgr_t temp;
	    cv::warpAffine(img, temp, trans, img.size());
	    img = temp;
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
	    std::string dFile = depthFiles[i];

	    readData(dFile, img);
	    
	    if(!noseg) // segment the image according to ground truth segmentation
	    {
		jp::img_id_t seg;
		getSegmentation(i, seg);
		
		for(unsigned x = 0; x < img.cols; x++)
		for(unsigned y = 0; y < img.rows; y++)
		    if(!seg(y, x)) img(y, x) = 0;
	    }
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
	    getBGR(i, img.bgr, noseg);
	    getDepth(i, img.depth, noseg);
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
	    jp::img_bgr_t segTemp;
	    readData(segFiles[i], segTemp);
	  
	    seg = jp::img_id_t::zeros(segTemp.rows, segTemp.cols);
	    int margin = 3; // in some of our rendered data there was a small artifact at the border - you can set this to zero if you have clean data
	    for(unsigned x = margin; x < seg.cols-margin; x++)
	    for(unsigned y = margin; y < seg.rows-margin; y++)
		if(segTemp(y,x)[0]) seg(y,x) =  1;
	}
        
        /**
         * @brief Get the ground truth Object Coordinate Label Images of the given frame.
         *
         * @param i Frame number.
         * @param img Output parameter. Segmentation mask.
         * @return void
         */
        void getObjCoordLabelImage(size_t i, jp::img_cordL_t & labelImg) const
        {
            std::string labelFile = objCoordLabelFiles[i];
            
            readData(labelFile, labelImg);
//            int a= (int)labelImg(284,278);
//            std::cout<<a<<std::endl;

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
	    readData(objFiles[i], img);
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
	    jp::img_depth_t imgDepth;
	    getDepth(i, imgDepth, true);
	    
	    img = jp::img_coord_t(imgDepth.rows, imgDepth.cols);
	    
	    #pragma omp parallel for
	    for(int x = 0; x < img.cols; x++)
	    for(int y = 0; y < img.rows; y++)
	    {
	       img(y, x) = pxToEye(x, y, imgDepth(y, x));
	    }
	}
	
    private:
	  
      /**
       * @brief Reads all file names in the various sub-folders of a dataset.
       * 
       * @param basePath Folder where all data sub folders lie.
       * @return void
       */
      void readFileNames(const std::string& basePath)
	{
	    std::cout << "Reading file names... " << std::endl;
	    std::string segPath = "/seg/", segSuf = ".png";
	    std::string bgrPath = "/rgb_noseg/", bgrSuf = ".png";
	    std::string dPath = "/depth_noseg/", dSuf = ".png";
	    std::string objPath = "/obj/", objSuf = ".png";
        std::string objLabelPath = "/objLabels/", objLabelSuf = ".png";

	    std::string infoPath = "/info/", infoSuf = ".txt";

	    bgrFiles = getFiles(basePath + bgrPath, bgrSuf);
	    depthFiles = getFiles(basePath + dPath, dSuf);
	    infoFiles = getFiles(basePath + infoPath, infoSuf, true);
	    segFiles = getFiles(basePath + segPath, segSuf, true);
	    objFiles = getFiles(basePath + objPath, objSuf, true);
        objCoordLabelFiles = getFiles(basePath+objLabelPath,objLabelSuf,true);
        
	}

	jp::id_t objID; // object ID this dataset belongs to
    
	// image data files
	std::vector<std::string> bgrFiles; // list of RGB files
	std::vector<std::string> depthFiles; // list of depth files
	// groundtruth data files
	std::vector<std::string> segFiles; // list of ground truth segmentation files	
	std::vector<std::string> objFiles; // list of ground truth object coordinate image files
	std::vector<std::string> infoFiles; // list of ground truth annotation files

    std::vector<std::string> objCoordLabelFiles; // list of ground truth object coordinate Label image files
    };
}

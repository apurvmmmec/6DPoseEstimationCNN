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
#include "multi_dataset.h"
#include <opencv2/highgui/highgui.hpp>
#include "thread_rand.h"

namespace jp
{
    /**
     * @brief Class that preloads and holds all data for training random forests.
     */
    class ImageCache
    {
    public:
        
        ImageCache() {}
        
        /**
         * @brief Reload data given object and background datasets. Also create pseudo-class labels for object coordinates.
         *
         * Depending on the parameters a random subset of images will be loaded per object or all images.
         * If all image of these datasets have been loaded before (and not just a fraction) the function will just recreate pseudo-class labels.
         *
         * @param dataset Dataset of object frames.
         * @param bgDataset Dataset of background images (negative class).
         * @return void
         */
        void reload(const jp::MultiDataset& dataset, const jp::Dataset& bgDataset)
        {
            GlobalProperties* gp = GlobalProperties::getInstance();
            
            if(gp->fP.maxImageCount > 0 || dataCache.empty()) // if this is false, all available images have been loaded before and reloading them is not necessary
            {
                dataCache.clear();
                depthCache.clear();
                gtCache.clear();
                objCache.clear();
                sampleSegCache.clear();
                sampleCounts.clear();
                idCache.clear();
                poseCache.clear();
                objProbCache.clear();
                
                bgPointer = 0;
                
                addSet(dataset);
                if(bgDataset.size() > 0)
                    addBGSet(bgDataset);
            }
            
            std::cout << "Recreating labelings ...";
            std::cout.flush();
            recreateLabels(); // create object coordinate pseudo class labels
            std::cout << " Done." << std::endl << std::endl;
        }
        
        /**
         * @brief Loads images and ground truth data. Also crops segmented images (to save memory).
         *
         * @param dataset Dataset of object frames.
         * @return void
         */
        void addSet(const jp::MultiDataset& dataset)
        {
            GlobalProperties* gp = GlobalProperties::getInstance();
            
            std::vector<unsigned> imgIdxList; // list of images to load
            std::vector<unsigned> objSamples; // list of samples to be drawn per image
            
            if(gp->fP.maxImageCount < 0) // just load all available images
            {
                for(unsigned imgIdx = 0; imgIdx < dataset.size(); imgIdx++)
                    imgIdxList.push_back(imgIdx);
                
                for(jp::id_t objID = 1; objID <= gp->fP.objectCount; objID++)
                    objSamples.push_back(gp->fP.trainingPixelsPerObject / dataset.size(objID));
            }
            else // load a maximum amount of training images per object (randomly choosen)
            {
                unsigned lowerBound = 0; // figure out which frame range of the combined dataset applies for the different objects
                
                for(jp::id_t objID = 1; objID <= gp->fP.objectCount; objID++)
                {
                    unsigned upperBound = lowerBound + dataset.size(objID);
                    
                    for(unsigned imgIdx = 0; imgIdx < gp->fP.maxImageCount; imgIdx++)
                        imgIdxList.push_back(irand(lowerBound, upperBound));
                    objSamples.push_back(gp->fP.trainingPixelsPerObject / gp->fP.maxImageCount);
                    
                    lowerBound = upperBound;
                }
            }
            
            std::cout << std::endl << "Loading dataset ..." << std::endl;
            
            for(auto imageIndex : imgIdxList)
            {
                bool useBG = false; // load segmented images (i.e. dont learn the background of training images)
                
                // load RGB and depth channel
                jp::img_bgrd_t image;
                dataset.getBGRD(imageIndex, image, useBG);
                
                // initialize discrete label groundtruth
                jp::img_label_t gt = jp::img_label_t::zeros(image.bgr.size()); // labels are filled in later with recreateLabels()
                
                // load segmentation mask
                jp::img_id_t seg;
                dataset.getSegmentation(imageIndex, seg);
                
                // load object coordinate ground truth
                jp::img_coord_t objPoints;
                dataset.getObj(imageIndex, objPoints);
                
                // load ground truth annotation
                jp::info_t info;
                dataset.getInfo(imageIndex, info);
                
                // determine that maximum simulated scale (when data augmentation is applied)
                float maxScale = (gp->fP.scaleRel)
                ? gp->fP.scaleMax / std::abs(info.center(2))
                : gp->fP.scaleMax;
                
                jp::img_id_t sampleSeg; // segmentation that is used for drawing training samples, its bigger then the normal segmentation to draw sample patch that show a fraction of the object although the center pixel is not on the object
                cropImages(image, gt, seg, objPoints, maxScale, sampleSeg); // crop the data according to the segmentation (cut away empty image parts to save memory)
                
                // combine data for the forest features
                jp::img_data_t dataItem;
                
                dataItem.seg = seg;
                dataItem.colorData = image.bgr;
                
                // initialize auto-context feature channels
                dataItem.labelData = std::vector<jp::img_label_t>(gp->fP.objectCount);
                dataItem.coordData = std::vector<jp::img_coord_t>(gp->fP.objectCount);
                
                for(unsigned o = 0; o < gp->fP.objectCount; o++)
                {
                    dataItem.labelData[o] = jp::img_label_t::zeros(seg.rows / gp->fP.acSubsample, seg.cols / gp->fP.acSubsample);
                    dataItem.coordData[o] = jp::img_coord_t::zeros(seg.rows / gp->fP.acSubsample, seg.cols / gp->fP.acSubsample);
                }
                
                dataCache.push_back(dataItem);
                if(gp->fP.useDepth) depthCache.push_back(image.depth); // depth is optional, omitted in RGB case
                gtCache.push_back(gt);
                objCache.push_back(objPoints);
                sampleCounts.push_back(objSamples[dataset.getObjID(imageIndex) - 1]);
                sampleSegCache.push_back(sampleSeg);
                idCache.push_back(dataset.getObjID(imageIndex));
                poseCache.push_back(info);
                
                // store object probability map for hard negative mining
                jp::img_stat_t initObjProb = jp::img_stat_t::zeros(seg.rows, seg.cols);
                for(unsigned x = 0; x < sampleSeg.cols; x++)
                    for(unsigned y = 0; y < sampleSeg.rows; y++)
                        if(sampleSeg(y, x)) initObjProb(y, x) = 1.f;
                
                objProbCache.push_back(initObjProb);
            }
            
            // push the background pointer back (marks where background images start)
            bgPointer = dataCache.size();
            std::cout << YELLOWTEXT(bgPointer << " images loaded.")
            << std::endl << std::endl;
            
            std::cout << "Samples per image: " << std::endl;
            for(unsigned i = 0; i < objSamples.size(); i++)
                std::cout << "Object " << i + 1 << ": " << objSamples[i] << std::endl;
            std::cout << std::endl;
        }
        
        /**
         * @brief Loads background images (negative class).
         *
         * @param bgDataset Dataset of background images (negative class).
         * @return void
         */
        void addBGSet(const jp::Dataset& bgDataset)
        {
            GlobalProperties* gp = GlobalProperties::getInstance();
            unsigned oldSize = dataCache.size(); // mark the point where the frames showing object end
            unsigned bgSamples = gp->fP.trainingPixelsPerObject / bgDataset.size() * gp->fP.trainingPixelFactorBG;
            
            std::cout << std::endl << "Loading background set ..." << std::endl;
            
            dataCache.resize(oldSize + bgDataset.size());
            if(gp->fP.useDepth) depthCache.resize(oldSize + bgDataset.size());
            sampleCounts.resize(oldSize + bgDataset.size());
            poseCache.resize(oldSize + bgDataset.size());
            objProbCache.resize(oldSize + bgDataset.size());
            sampleSegCache.resize(oldSize + bgDataset.size());
            idCache.resize(oldSize + bgDataset.size());
            
#pragma omp parallel for
            for(unsigned imageIndex = 0; imageIndex < bgDataset.size(); imageIndex++)
            {
                // load RGB and depth channels
                jp::img_bgrd_t image;
                bgDataset.getBGRD(imageIndex, image, true);
                
                // segmentation is the complete image
                jp::img_id_t seg = jp::img_id_t::ones(image.bgr.rows, image.bgr.cols);
                
                dataCache[oldSize + imageIndex].seg = seg;
                dataCache[oldSize + imageIndex].colorData = image.bgr;
                
                // initialize object coordinate feature channels
                dataCache[oldSize + imageIndex].labelData = std::vector<jp::img_label_t>(gp->fP.objectCount);
                dataCache[oldSize + imageIndex].coordData = std::vector<jp::img_coord_t>(gp->fP.objectCount);
                
                for(unsigned o = 0; o < gp->fP.objectCount; o++)
                {
                    dataCache[oldSize + imageIndex].labelData[o] = jp::img_label_t::zeros(seg.rows / gp->fP.acSubsample, seg.cols / gp->fP.acSubsample);
                    dataCache[oldSize + imageIndex].coordData[o] = jp::img_coord_t::zeros(seg.rows / gp->fP.acSubsample, seg.cols / gp->fP.acSubsample);
                }
                
                if(gp->fP.useDepth) depthCache[oldSize + imageIndex] = image.depth; // depth is optional, ommited in RGB case
                sampleCounts[oldSize + imageIndex] = bgSamples;
                sampleSegCache[oldSize + imageIndex] = seg;
                poseCache[oldSize + imageIndex] = info_t(); // no ground truth for background images
                objProbCache[oldSize + imageIndex] = jp::img_stat_t::ones(seg.rows, seg.cols); // initialize probability map for hard negative mining
                idCache[oldSize + imageIndex] = 0;
            }
            
            std::cout << YELLOWTEXT(bgDataset.size() << " background images loaded. (" << dataCache.size() << " total)")
            << std::endl << std::endl;
            
            std::cout << "Samples per image: " << std::endl;
            std::cout << "Background: " << bgSamples << std::endl << std::endl;
        }
        
        /**
         * @brief Generates discrete label images for each frame by randomly clustering object coordinates.
         
         * @return void
         */
        void recreateLabels() const
        {
            // find new cluster centers per object
            std::vector<std::vector<jp::coord3_t>> objClusterCenters;
            getRandomClusterCenters(objClusterCenters);
            
            // create a label image for each frame
#pragma omp parallel for
            for(unsigned imgIdx = 0; imgIdx < objCache.size(); imgIdx++)
            {
                // initialize label image (areas outside segmentation get label zero)
                jp::img_label_t labelImg = jp::img_label_t::zeros(objCache[imgIdx].rows, objCache[imgIdx].cols);
                
                for(unsigned x = 0; x < objCache[imgIdx].cols; x++)
                    for(unsigned y = 0; y < objCache[imgIdx].rows; y++)
                    {
                        if(!dataCache[imgIdx].seg(y, x)) continue; // skip areas outside segmentation
                        
                        jp::coord3_t coord = objCache[imgIdx](y, x);
                        if(!onObj(coord)) continue; // skip areas outside segmentation
                        
                        jp::id_t objID = idCache[imgIdx];
                        
                        // find nearest cluster center for current object coordinate
                        float minDist = -1;
                        unsigned minCluster;
                        
                        for(unsigned clusterIdx = 0; clusterIdx < objClusterCenters[objID-1].size(); clusterIdx++)
                        {
                            float dist = cv::norm(objClusterCenters[objID-1][clusterIdx], coord);
                            if(minDist < 0 || dist < minDist)
                            {
                                minDist = dist;
                                minCluster = clusterIdx;
                            }
                        }
                        
                        // store global label from local cluster label and object ID
                        labelImg(y, x) = getLabel(objID, minCluster+1);
                    }
                gtCache[imgIdx] = labelImg; // overwrites previous label image
            }
        }
        
        std::vector<jp::img_data_t> dataCache; // input data the forest features operate on (RGB, segmentation, auto-context feature channels)
        std::vector<jp::img_depth_t> depthCache; // depth channels (might be empty in RGB case)
        mutable std::vector<jp::img_label_t> gtCache; // discrete label images used to train the forest structure (object coordinate pseudo classes)
        std::vector<jp::img_coord_t> objCache; // ground truth object coordinate images used to train the leaf distributions
        std::vector<jp::img_id_t> sampleSegCache; // segmentation mask which tell where to draw samples (also outside the object if the patch still covers parts of the object)
        std::vector<unsigned> sampleCounts; // how many samples to draw per image
        std::vector<jp::id_t> idCache; // ID of the object associated with each image
        std::vector<jp::info_t> poseCache; // ground truth pose information
        std::vector<jp::img_stat_t> objProbCache; // probability maps used in hard negative mining
        
        unsigned bgPointer; // marks the frame number where object frames end and background frames begin
        
    private:
        
        /**
         * @brief Selects random object coordinates per object which are used as seeds for object coordinate pseudo classes.
         *
         * @param objClusterCenters Output parameter. For each object a list of cluster centeres (in 3D).
         * @return void
         */
        void getRandomClusterCenters(std::vector<std::vector<jp::coord3_t>>& objClusterCenters) const
        {
            GlobalProperties* gp = GlobalProperties::getInstance();
            objClusterCenters.resize(gp->fP.objectCount);
            
            // order images by objects
            std::vector<std::vector<unsigned>> objLists(gp->fP.objectCount);
            for(unsigned imgIdx = 0; imgIdx < bgPointer; imgIdx++)
                objLists[idCache[imgIdx]-1].push_back(imgIdx);
            
            int labelCount = gp->fP.getCellCount(); // how many cluster centers per object?
            
#pragma omp parallel for
            for(unsigned objIdx = 0; objIdx < objLists.size(); objIdx++)
            {
                for(unsigned i = 0; i < labelCount; i++)
                {
                    bool found = false; // repeat until a valid object coordinate was choosen
                    while(!found)
                    {
                        // choose a random frame for the current object and a random pixel location
                        unsigned imgIdx = irand(0, objLists[objIdx].size());
                        imgIdx = objLists[objIdx][imgIdx];
                        unsigned x = irand(0, dataCache[imgIdx].seg.cols);
                        unsigned y = irand(0, dataCache[imgIdx].seg.rows);
                        if(dataCache[imgIdx].seg(y, x) && onObj(objCache[imgIdx](y, x))) // repeat if the random pixel hit the background
                        {
                            found = true;
                            objClusterCenters[objIdx].push_back(objCache[imgIdx](y, x));
                        }
                    }
                }
            }
        }
        
        /**
         * @brief Removes empty parts of the current frame to save memory. It also produces the sample segmentation by blowing original segmentation a bit.
         * 
         * @param image Output parameter. RGB and depth channel.
         * @param gt Output parameter. Discrete ground truth labels (object coordinate pseudo classes).
         * @param seg Output parameter. Ground truth segmentation.
         * @param objPoints Output parameter. Ground truth object coordinates.
         * @param maxScale Maximum scale that may be applied during data augmentation. The images are cropped in a way that (scaled) features do not reach outside its border.
         * @param sampleSeg Output parameter. Segmentation for drawing training samples. Generated in this method.
         * @return void
         */
        void cropImages(
                        jp::img_bgrd_t& image, 
                        jp::img_label_t& gt,
                        jp::img_id_t& seg, 
                        jp::img_coord_t& objPoints,
                        float maxScale,
                        jp::img_id_t& sampleSeg)
        {
            int minX, minY, maxX, maxY;
            maxX = maxY = 0;
            minX = seg.cols;
            minY = seg.rows;
            
            //determine bounding box and max min depth
            for(int x = 0; x < seg.cols; x++)
                for(int y = 0; y < seg.rows; y++)
                {
                    if(seg(y, x))
                    {
                        maxX = std::max(x, maxX);
                        minX = std::min(x, minX);
                        maxY = std::max(y, maxY);
                        minY = std::min(y, minY);
                    }
                }
            
            // add margin for the max possible feature range
            int margin = 0.5 + GlobalProperties::getInstance()->fP.maxOffset * maxScale;
            maxX += margin;
            maxY += margin;
            minX -= margin;
            minY -= margin;
            
            // ensure bb within image area
            maxX = clamp(maxX, 0, seg.cols - 1);
            minX = clamp(minX, 0, seg.cols - 1);
            maxY = clamp(maxY, 0, seg.rows - 1);
            minY = clamp(minY, 0, seg.rows - 1);
            
            // crop the data
            cv::Rect bb(minX, minY, maxX - minX, maxY - minY);
            image.bgr = image.bgr(bb).clone();
            image.depth = image.depth(bb).clone();
            gt = gt(bb).clone();
            seg = seg(bb).clone();
            objPoints = objPoints(bb).clone();
            
            // copy the segmentation and blow it up (samples may be drawn outside the object as long as the patch still shows parts of the object)
            sampleSeg = seg.clone();
            
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(margin, margin));
            cv::dilate(sampleSeg, sampleSeg, kernel);
        }
    };
}

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

#include "dataset.h"
#include "Hypothesis.h"

namespace jp
{
    jp::label_t getLabel(jp::id_t objID, jp::cell_t objCell)
    {
	// maping of global labels to object ID and local labels as follows:
	// map 0		to 0:0
	// map 1		to 1:1
	// map cellCount	to 1:cellCount
	// map cellCount+1 	to 2:1
	// and so on...      
	if(objID < 1) throw std::runtime_error("Invalid object passed (ID < 1)");
	jp::cell_t cellCount = GlobalProperties::getInstance()->fP.getCellCount();
	return (objID - 1) * cellCount + objCell;
    }
    
    jp::coord3_t pxToEye(int x, int y, jp::depth_t depth)
    {
	jp::coord3_t eye;
	
	if(depth == 0) // depth hole -> no camera coordinate
	{
	    eye(0) = 0;
	    eye(1) = 0;
	    eye(2) = 0;
	    return eye;
	}
	
	GlobalProperties* gp = GlobalProperties::getInstance();
	
	eye(0) = (short) ((x - (gp->fP.imageWidth / 2.f + gp->fP.xShift)) / (gp->fP.focalLength / depth));
	eye(1) = (short) -((y - (gp->fP.imageHeight / 2.f + gp->fP.yShift)) / (gp->fP.focalLength / depth));
	eye(2) = (short) -depth; // camera looks in negative z direction
	
	return eye;
    }

    bool onObj(const jp::coord3_t& pt)
    {
	return ((pt(0) != 0) || (pt(1) != 0) || (pt(2) != 0));
    }

    bool onObj(const jp::label_t& pt)
    {
	return (pt != 0);
    }
}

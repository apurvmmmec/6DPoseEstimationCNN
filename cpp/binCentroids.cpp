//
//  binCentroids.cpp
//  jointpose
//
//  Created by Apurv Nigam on 14/08/2017.
//
//

#include <iostream>
#include "binCentroids.h"

using namespace std;

vector<cv::Point3f> createCentVector(){
    
    int bin_num=125;
    vector<cv::Point3f> binCentroids;
    for(int i=0;i<bin_num*3;i++){
        float x= *(binArray+i);
        i++;
        float y =*(binArray+i);
        i++;
        float z =*(binArray+i);
        
          
        cv::Point3f tempPt(x,y,z);
        binCentroids.push_back(tempPt);
        //    cout<<tempPt<<endl;
    }
    
    return binCentroids;
    
    
}

import numpy as np

def getBB3D(maxExtent):

    bbminX = -maxExtent[0]/2;
    bbmaxX =  maxExtent[0]/2;
    
    bbminY = -maxExtent[1]/2;
    bbmaxY =  maxExtent[1]/2;
    
    bbminZ = -maxExtent[2]/2;
    bbmaxZ =  maxExtent[2]/2;

    bbObjCord = np.array([
         bbminX, bbminY, bbminZ,
         bbmaxX, bbminY, bbminZ,
         bbmaxX, bbmaxY, bbminZ,
         bbminX, bbmaxY, bbminZ,
         bbminX, bbminY, bbmaxZ,
         bbmaxX, bbminY, bbmaxZ,
         bbmaxX, bbmaxY, bbmaxZ,
         bbminX, bbmaxY, bbmaxZ]).reshape([8,3]);
    
    return bbObjCord
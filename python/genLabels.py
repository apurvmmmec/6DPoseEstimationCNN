import numpy as np

def getLabels(bin_Dim):
    labels=np.zeros([pow(bin_Dim,3),1],dtype=np.uint16);
    ct=0;
    for k in range(1,bin_Dim+1):
        for i in range(1,bin_Dim+1):
            for j in range(1,bin_Dim+1):
                labels[ct,0]= int('%d%d%d'%(i,j,k))
                ct = ct +1;

    return labels

# labels = getLabels()
# print labels
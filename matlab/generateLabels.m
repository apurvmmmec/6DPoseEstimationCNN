function [labels] = generateLabels(binDim)
labels=zeros(power(binDim,3),1);
ct=0;
for k=1:binDim
    for i=1:binDim
        for j=1:binDim
            ct=ct+1;
           labels(ct,1)= str2num( sprintf('%d%d%d',i,j,k)) ;           
        end
    end
end

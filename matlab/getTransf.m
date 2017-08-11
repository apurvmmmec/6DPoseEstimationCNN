function [xform] = getTransf(num)
fname = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/poses/Can/info_%05d.txt',num);
fid = fopen(fname);
tline = fgetl(fid);
i=0;
r= zeros(3,3);
t=zeros(3,1);
while ischar(tline)
    i=i+1;
    C = strsplit(tline,' ');
    
    if(i==5)
        r(1,1) = str2double(C(1));
        r(1,2) = str2double(C(2));
        r(1,3) = str2double(C(3));
    elseif(i==6)
            r(2,1) = str2double(C(1));
        r(2,2) = str2double(C(2));
        r(2,3) = str2double(C(3));
    elseif(i==7)
        r(3,1) = str2double(C(1));
        r(3,2) = str2double(C(2));
        r(3,3) = str2double(C(3));
    elseif(i==9)
        t(1,1) = str2double(C(1));
        t(2,1) = str2double(C(2));
        t(3,1) = str2double(C(3)); 
    end
    tline = fgetl(fid);  
end

xform = [r  t; 0 0 0 1];
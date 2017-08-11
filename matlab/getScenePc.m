function [pc, colorMap] = getScenePc(objId,imId)

depth=imread(sprintf('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/%02d/depth/%04d.png',objId,imId));
color=im2double(imread(sprintf('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/%02d/rgb/%04d.png',objId,imId)));
seg=imread(sprintf('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/%02d/seg/%04d.png',objId,imId));

% toc=getTransf(imId);
r=[ 0.20165101 -0.97653198  0.0756456 ;
    -0.93756402 -0.21479499 -0.273563  ;
    0.283391   -0.0157584  -0.958875  ];
t=[-0.00556719;
    -0.05020958;
    0.67473655];
toc = [r t;0 0 0 1]

%     objCal = zeros(size(seg,1),size(seg,2),4);
% maxExtent = [0.0775993; 0.0917691; 0.0758686]
% maxExtent = [0.127633 ;0.117457; 0.0670107];
%     maxExtent = [0.129444; 0.0982834 ;0.109656];
%     maxExtent= [0.181796 ;0.193734; 0.100792];
%     objC=zeros(1,3);
numPt=0;
pc = zeros(307200,3);
colorMap = zeros(307200,3);

for x=1:640
    for y=1:480
        %                                 if(seg(y,x) >0)
        numPt=numPt+1;
        d = depth(y,x);
        xc=(double(x)-325.2611)*(double(d)/(1000.0*572.4114));
        yc=(y-242.04899)*(double(d)/(1000*573.57043));
        zc=double(d)/1000;
        
        camC=[xc;yc;zc;1];
        
        xo = (inv(toc)*camC);
        
        pc(numPt,1) = xo(1,1);
        pc(numPt,2) = xo(2,1);
        pc(numPt,3) = xo(3,1);
        colorMap(numPt,1) = color(y,x,1);
        colorMap(numPt,2) = color(y,x,2);
        
        colorMap(numPt,3) = color(y,x,3);
    end
    
end


%     imshow(objCal);
% figure


end








close all;
clear all;
objId=5;
binDim =5;

%CAmera Intrinsic Param Matrix
camMat = [572.41140   0          325.2611;
    0           573.57043  242.04899;
    0           0          1];
cam = cameraParameters('IntrinsicMatrix', camMat')

for n=30:30
    n
    camCordDisp=0;
    maxExtent= [0.100792;0.181796 ;0.193734 ];
    
    bb3D = getBB3D(maxExtent);
    
    
    [pc,colorMap] = reconstructScene(objId,n);
    
    orpc = pointCloud(pc,'Color',colorMap);
    figure
    pcshow(orpc,'VerticalAxis','Y', 'VerticalAxisDir','Up');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(0,0)
    hold on
    bb3D = getBB3D(maxExtent);
    
    r=[ 0.97441399  0.18984599 -0.12032   ;
        -0.0544436  -0.320014   -0.94584697;
        -0.218069    0.92819703 -0.30149001];
    t=[  121.14949491;
        -113.72310242;
        1030.73827876]/1000;
    
    
    r=[ 0.86705204 -0.49718702  0.03202842;
        0.24091294  0.47466612  0.84655362;
        -0.43609828 -0.72628999  0.53133902];
    t=[-0.17207284;
        0.0797642 ;
        -1.08804289];
    
   r=[ 0.96430065  0.01840398 -0.26416955;
 -0.24385964 -0.32717404 -0.91295653;
 -0.10323145  0.94478486 -0.31100615];
t=[ 0.12146543;
 -0.1136339 ;
  1.03314563];
    

r=[ 0.32940261 -0.8400847  -0.43098911;
 -0.40457444 -0.53802011  0.73949569;
 -0.85311983 -0.06922463 -0.51710202];
t=[-0.09291715;
  0.08561312;
 -0.78881791];
    toc = [r t;0 0 0 1];
    t0 = [0;0;0];
    toc1 = [rotx(180) t0;0 0 0 1];
    
    
    bbHomo = [bb3D' ;ones(1,8)];
    bb3DCamCordH = (toc1)*(toc)*bbHomo;
    bb3DCamCord = bb3DCamCordH(1:3,:)./bb3DCamCordH(4,:);
    bb3D =bb3DCamCord';
    
    
    drawBB3D(bb3D)
    
end

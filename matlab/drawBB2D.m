function drawBB2D()

close all;
clear all;
objId=5;
binDim =5;

%CAmera Intrinsic Param Matrix
camMat = [572.41140   0          325.2611;
    0           573.57043  242.04899;
    0           0          1];
cam = cameraParameters('IntrinsicMatrix', camMat')

n=100
% maxExtent= [ 0.21567000000;
%     0.12185570000 ;
%     0.21941000000 ];

    maxExtent= [0.100792;0.181796 ;0.193734 ];

%Read Depth, Color and sgmentation mask
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
depth=imread(sprintf('%stest/%02d/depth/%04d.png',base_path,objId,n));
color=im2double(imread(sprintf('%stest/%02d/rgb/%04d.png',base_path,objId,n)));

r=[ -0.63901401 0.75430501 -0.15061900;
    0.67842001 0.46040899 -0.57251197;
    -0.36250201 -0.46802601 -0.80594301]
t=[0.11130890699;
    -0.10627223712;
    0.97351646174];

%Can 100
 r=[ 0.89526898  0.37082601 -0.246943  ;
 -0.113724   -0.34570301 -0.931427  ;
 -0.43076599  0.86196202 -0.26732501];
t=[ -100.50654196;
    89.42205556;
  1038.79488726]/1000
toc = [r t;0 0 0 1];
t0 = [0;0;0];
toc1 = [rotx(180) t0;0 0 0 1];

% Get the 3D boinding box
bb3D = getBB3D(maxExtent);

bbHomo = [bb3D' ;ones(1,8)];
bb3DCamCordH = (toc)*bbHomo;
bb3DCamCord = bb3DCamCordH(1:3,:)./bb3DCamCordH(4,:);
bb3D =bb3DCamCord';

bb2dH = camMat*bb3D'
bb2d = bb2dH(1:2,:)./bb2dH(3,:)
bb2D = bb2d'


imshow(color)
hold on
v1 = bb2D(1,1:2);
v2 = bb2D(2,1:2);
v3 = bb2D(3,1:2);
v4 = bb2D(4,1:2);
v5 = bb2D(5,1:2);
v6 = bb2D(6,1:2);
v7 = bb2D(7,1:2);
v8 = bb2D(8,1:2);
v=[v2;v1];
plot(v(:,1),v(:,2),'r','LineWidth',3)
hold on
v=[v2;v3];
plot(v(:,1),v(:,2),'r','LineWidth',3)
hold on
v=[v3;v4];
plot(v(:,1),v(:,2),'r','LineWidth',3)
hold on
v=[v1;v4];
plot(v(:,1),v(:,2),'r','LineWidth',3)
hold on
v=[v1;v5];
plot(v(:,1),v(:,2),'r','LineWidth',3)
hold on
v=[v2;v6];
plot(v(:,1),v(:,2),'r','LineWidth',3)
hold on
v=[v3;v7];
plot(v(:,1),v(:,2),'r','LineWidth',3)
hold on
v=[v4;v8];
plot(v(:,1),v(:,2),'r','LineWidth',3)

hold on
v=[v5;v6];
plot(v(:,1),v(:,2),'r','LineWidth',3)


hold on
v=[v6;v7];
plot(v(:,1),v(:,2),'r','LineWidth',3)

hold on
v=[v7;v8];
plot(v(:,1),v(:,2),'r','LineWidth',3)

hold on
v=[v5;v8];
plot(v(:,1),v(:,2),'r','LineWidth',3)

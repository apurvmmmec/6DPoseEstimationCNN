close all;
clear all;
depth=imread('./test/Accv_Cat/depth_noseg/depth_00100.png');
color=im2double(imread('./test/Accv_Cat/rgb_noseg/color_00100.png'));

obj=(imread('test/Accv_Cat/obj/obj_00100.png'));
seg = imread('test/Accv_Cat/seg/seg00100.png');
w=640;
h=420;
% toc= [-0.7441   -0.1650    0.6474    0.1663;
%     0.4457    0.5992    0.6651   -0.0876;
%     -0.4977    0.7834   -0.3723   -0.9109;
%     0         0         0    1.0000];
r=[-0.49824 -0.37171 0.783422;
0.648946 0.439464 0.621212;
-0.575142 0.817844 0.0222681];
t= [0.43754; 0.247973 ;-1.09042];

toc=[r t;0 0 0 1];
objCal = zeros(size(obj));
objCordRaw = zeros(size(obj));

pc = zeros(307200,3);
    colorMap = zeros(307200,3);
% maxExtent = [0.0775993; 0.0917691; 0.0758686]
maxExtent = [0.127633 ;0.117457; 0.0670107];
numPt=0;
for x=1:640
    for y=1:480
        %         x=435
        %         y=300
        if(seg(y,x) >0)
            numPt=numPt+1;
            d = depth(y,x);
            xc=(double(x)-320.0)*(double(d)/(1000.0*575.816));
            yc=-(y-240)*(double(d)/(1000*575.816));
            zc=-double(d)/1000;
            
            camC=[xc;yc;zc;1];
            xo = (inv(toc)*camC);
%             objCordRaw(y,x,1) = camC(1,1);
%             objCordRaw(y,x,2) = camC(2,1);
%             objCordRaw(y,x,3) = camC(3,1);

            objCal(y,x,3) = xo(3,1)+maxExtent(3)/2;
            objCal(y,x,3) = (objCal(y,x,3)/maxExtent(3));
            
            objCal(y,x,2) = xo(2,1)+maxExtent(2)/2;
            objCal(y,x,2) = (objCal(y,x,2)/maxExtent(2));
            
            objCal(y,x,1) = xo(1,1)+maxExtent(1)/2;
            objCal(y,x,1) = (objCal(y,x,1)/maxExtent(1));
            
%             coord = double(obj(y,x,1)+(maxExtent(1)/2))
%             obj(y,x,1) = (coord/(maxExtent(1)))*255;
%             
%             coord = double(obj(y,x,2)+(maxExtent(2)/2))
%             obj(y,x,2) = (coord/(maxExtent(2)))*255;
%             
%             coord = double(obj(y,x,3)+(maxExtent(3)/2))
%             obj(y,x,3) = (coord/(maxExtent(3)))*255;
            
            
            
            
            pc(numPt,1) = objCal(y,x,1);
            pc(numPt,2) = objCal(y,x,2);
            pc(numPt,3) = objCal(y,x,3);
            colorMap(numPt,1) = color(y,x,1);
            colorMap(numPt,2) = color(y,x,2);
            
            colorMap(numPt,3) = color(y,x,3);
               
        end
    end
    
end
figure
    orpc = pointCloud(pc,'Color',colorMap);
    
    
    pcshow(orpc);
    xlabel('X');
    ylabel('Y');
    zlabel('Z')
   
figure
imshow(objCal);
% figure;
% imshow((obj))


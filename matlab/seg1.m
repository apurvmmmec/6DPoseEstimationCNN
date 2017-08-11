close all
clear all
ptCloud = pcread('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/models/obj_09.ply');
pcshow(ptCloud);
xlabel('X');
ylabel('Y');
zlabel('Z');
% num = ptCloud.Count;

for n=6
img = zeros(480,640);


camMat = [572.41140   0          325.2611     0;
    0           573.57043  242.04899    0;
    0           0          1            0];



% r= [-0.59296 -0.464535 0.65807;
%     0.411043 0.528291 0.743223;
%     -0.692743 0.711035 -0.12226];
% t= [0.411636 ;0.240788; -1.12979];
% r=[0.0950661 -0.983309 0.155129; 
% 0.741596 0.173913 0.647911 ;
% -0.664076 0.0534489 0.745752];
% t= [6.6199 ;
% -11.628 ;
% 108.777 ]/100;
% 
% maxExtent = [0.127633 ;0.117457; 0.0670107];
% t_model = [maxExtent(1,1)/2;maxExtent(2,1)/2;maxExtent(3,1)/2];
% r_model= roty(180);
% r_model= r_model*rotz(180);
% r_model_inv= inv(r_model);
% r1 = r*r_model_inv;
% t = t + r1* (r_model*t_model);
% toc = [r1 t;0 0 0 1];
toc = getTransf(n);

% num = ptCloud.Count
pc = zeros(15736,3);
% for i=1:num
fid = fopen('../../../hinterstoisser/OcclusionChallengeICCV2015/models/Cat/005.xyz');
% fid = fopen('pc2.xyz');

tline = fgetl(fid);
numPts=0;
% i=0;
while ischar(tline)
    i=i+1;
    C = strsplit(tline,' ');
    
    if(i==1)
        numPts= C(1);
        pc = zeros(str2double(C(1)),3);
        %         objCordRaw = zeros(str2double(C(1)),3);
        
    else
        x = str2double(C(1))/1;
        y = str2double(C(2))/1;
        z = str2double(C(3))/1;
        %     x = ptCloud.Location(i,1)/1000;
        %     y = ptCloud.Location(i,2)/1000;
        %     z = ptCloud.Location(i,3)/1000;
        
%                 pc(i,1)=x;
%                 pc(i,2)=y;
%                 pc(i,3)=z;
        %         tf = toc*[x;y;z;1];
%         camCord = (toc)*[-y;-z;x;1];
                camCord = (toc)*[x;y;z;1];

        
        xc = camCord(1,1);
        yc = camCord(2,1);
        zc = camCord(3,1);
        %         objCordRaw(i-1,1)=tf(1,1);
        %         objCordRaw(i-1,2)=tf(2,1);
        %         objCordRaw(i-1,3)=tf(3,1);
        d  = -zc;
        ptx = ((xc*(572.41140)/d)+325.2611);
        pty = (-(yc*(573.57043)/d)+242.0489);
        if((pty<480) && (pty >=1)) && ((ptx<640) &&(ptx >=1))
            img(round(pty),round(ptx))=1;
        end
    end
    tline = fgetl(fid);
    
end
% orpc = pointCloud(pc);

% pcshow(orpc);
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% figure
% imshow(img);
% fname = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/seg/Cat/seg%05d.jpg',n)
% imwrite(img,fname);
end

close all

color=im2double(imread('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/09/rgb/0683.png'));
imshow(color);
camMat = [572.41140   0          325.2611;
    0           573.57043  242.04899;
    0           0          1];

% K=[531 0 320;
%     0  531 240;
%     0 0 1];

ptCloud = pcread('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/models/obj_09.ply');
figure
pcshow(ptCloud);
xlabel('X');
ylabel('Y');
zlabel('Z');

fid = fopen('../../../hinterstoisser/OcclusionChallengeICCV2015/models/Duck/007.xyz');
tline = fgetl(fid);
obj = zeros(1,3);
i=0;
while ischar(tline)
    i=i+1;
    C = strsplit(tline,' ');
    
    obj(i,1) = str2double(C(1))*1;
    obj(i,2) = str2double(C(2))*1;
    obj(i,3) = str2double(C(3))*1;
    
    tline = fgetl(fid);
end
fclose(fid);
ptCloud = pointCloud(obj);
figure
pcshow(ptCloud);
xlabel('X');
ylabel('Y');
zlabel('Z');
obj=obj*rotx(90)*rotz(90)
ptCloud = pointCloud(obj);
figure
pcshow(ptCloud);
xlabel('X');
ylabel('Y');
zlabel('Z');

num = ptCloud.Count;
% r=[0.1133    0.9914    0.0661;
%     0.5854   -0.0129   -0.8106;
%     -0.8028    0.1305   -0.5818];
% t=[-13.0792;
%     -7.83575;
%     104.177]*10;

% r= [0.98378801, -0.04328500, -0.17403300;
%     -0.16262101, -0.62442100, -0.76397198;
%     -0.07560160, 0.77988797, -0.62133700];
% t=[61.25955137; -91.60362203; 1032.82310720];
% 
% r= [ 0.98310202 -0.0472039  -0.17686699;
%  -0.167686   -0.61976397 -0.76666403;
%  -0.0734263   0.78336698 -0.61720699];
% t=[   55.44819932;
%    -81.89323636;
%   1025.71399065];

r=[ 0.80779658  0.41842626  0.41535167;
 -0.58285539  0.67293313  0.45560313;
 -0.08885524 -0.61009152  0.78741073];
t=[  438.18869296;
  -245.03298065;
  1083.9189668] ;

r=[ 0.20165101 -0.97653198  0.0756456;
 -0.93756402 -0.21479499 -0.273563;
  0.283391   -0.0157584  -0.958875];
t=[-0.00556719;
 -0.05020958;
 0.67473655];

% t= [-0.073782; -0.030049 ;-0.993205]

toc1 = [r*roty(180)*rotz(180) t];

% for n=11:100
% n=1;
% fid = fopen('../../../hinterstoisser/OcclusionChallengeICCV2015/models/Ape/001.xyz');

% toc1=getTransf(n);
img = zeros(480,640,'uint8');

% tline = fgetl(fid);
numPts=size(ptCloud.Location,1);
i=0;
% pc = zeros(1,3);

P = camMat*toc1;
pts = ptCloud.Location;
pts = obj;
pts_h = [pts ones(numPts,1)];
pts_im = P*pts_h';

pts_im=round(pts_im./pts_im(3,:));
% pts_3d = pts*roty(180);
% 
% pts_3d = pts_3d*toc1;
% % pts_3d = pts_3d./pts_3d(:,4);
% pts_3d = pts_3d(:,1:3);

for i=1:numPts
    if( (pts_im(2,i) >0) && (pts_im(2,i) <480) && (pts_im(1,i) >0) && (pts_im(1,i) <640))
        color(pts_im(2,i),pts_im(1,i),:)=0; 
    end
end
P;
% while ischar(tline)
%     i=i+1;
%     C = strsplit(tline,' ');
%     
%     
%     x = str2double(C(1))/1;
%     y = str2double(C(2))/1;
%     z = str2double(C(3))/1;
%     
%     camCord = (toc1)*[x;y;z;1];
%     xc = camCord(1,1);
%     yc = camCord(2,1);
%     zc = camCord(3,1);
%     
%     d  =zc;
%     ptx = ((xc*(572.41140)/d)+320);
%     pty = ((yc*(573.57043)/d)+240);
%     if((pty<480) && (pty >=0)) && ((ptx<640) &&(ptx >=0))
%         img(ceil(pty),ceil(ptx))=255;
%     end
%     %     end
%     
%     tline = fgetl(fid);
% end

% fclose(fid);
% figure;
% orpc = pointCloud(pc);
% pcshow(orpc);
% xlabel('X');
% ylabel('Y');
% zlabel('Z');

figure
imshow(color);

% figure
% orpc = pointCloud(obj);
% pcshow(orpc);
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% fname = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/seg/Can/seg%05d.bmp',n);
% img=im2double(img);
% imwrite(img,fname);

% end



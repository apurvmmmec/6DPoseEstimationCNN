close all

camMat = [572.41140   0          325.2611     0;
          0           573.57043  242.04899    0;
          0           0          1            0];

% r=[0.812471 -0.0092161 -0.582957;
% -0.425033 0.675072 -0.603038;
% 0.399088 0.737715 0.544548];
% t= [-0.073782; -0.030049 ;-0.993205]

% toc1 = [r t;0 0 0 1];

for n=1:1
    fid = fopen('../../../hinterstoisser/OcclusionChallengeICCV2015/models/Can/004.xyz');

toc1=getTransf(n);
img = zeros(480,640,'uint8');

tline = fgetl(fid);
numPts=0;
i=0;
        pc = zeros(1,3);

while ischar(tline)
    i=i+1;
    C = strsplit(tline,' ');
    
%     if(i==1)
%         numPts= C(1);
%     else
        x = str2double(C(1))/1;
        y = str2double(C(2))/1;
        z = str2double(C(3))/1;
%         pc(i,1)=x;
%         pc(i,2)=y;
%         pc(i,3)=z;

%         pc(i-1,1)=x;
%         pc(i-1,2)=y;
%         pc(i-1,3)=z;      
%         pt = camMat*(toc)*[x;y;z;1];   
%         ptx = pt(1,1)/pt(3,1);
%         pty = pt(2,1)/pt(3,1);
        
         camCord = (toc1)*[x;y;z;1];
         xc = camCord(1,1);
         yc = camCord(2,1);
         zc = camCord(3,1);
         
         d  =- zc;
         ptx = ((xc*(572.41140)/d)+320);
         pty = (-(yc*(573.57043)/d)+240);
        if((pty<480) && (pty >=0)) && ((ptx<640) &&(ptx >=0))
            img(ceil(pty),ceil(ptx))=255;
        end
%     end
    
    tline = fgetl(fid);
end

fclose(fid);
% figure;
% orpc = pointCloud(pc);
% pcshow(orpc);
% xlabel('X');
% ylabel('Y');
% zlabel('Z');

% figure
% imshow(img);
fname = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/seg/Can/seg%05d.bmp',n);
img=im2double(img);
imwrite(img,fname);

end



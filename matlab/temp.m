close all
clear all
n=151

depth=imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/RGB-D/depth_noseg/depth_%05d.png',n));
    color=im2double(imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/RGB-D/rgb_noseg/color_%05d.png',n)));
    seg = imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/seg/Can/seg%05d.bmp',n));

    img=zeros(480,640);
    
    for x=1:640
        for y=1:480
            if(seg(y,x) >0)
                d = depth(y,x);
                img(y,x)=d;
                
            end
        end
    end
    
    imshow(img,[]);
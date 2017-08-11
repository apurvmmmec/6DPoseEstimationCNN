for n=0:0
    
    %Read Depth, Color and sgmentation mask
%     depth=imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/RGB-D/depth_noseg/depth_%05d.png',n));
    color=im2double(imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/RGB-D/rgb_noseg/color_%05d.png',n)));
    labelImg = (imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/objCordLabel/Can/label_%05d.png',n)));
%     seg = imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/seg/Can/seg%05d.bmp',n));
    ct=0;
    for r = 40:10:380
        for c= 40:10:540
            ct=ct+1;
            subImg = color(r:r+255,c:c+63,:);
            subName=sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/subsampled/000011/sub_%05d.png',ct);
            [w h ch]= size(subImg)
%             r+32
%             c+32
            label = labelImg(r+32,c+32)
            if(label>0)
                label
            end
%             imwrite(subImg,subName);
%             imshow(subImg);
            
        end
    end
    
    
    
end

% for i=0:125
%     dirName = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/LabeledImg/Can/%03d/',i)
%     mkdir(dirName);
% end
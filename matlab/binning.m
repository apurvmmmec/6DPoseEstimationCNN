close all;
clear all;
objId=5;
binDim =4;

%CAmera Intrinsic Param Matrix
camMat = [572.41140   0          325.2611;
    0           573.57043  242.04899;
    0           0          1];
cam = cameraParameters('IntrinsicMatrix', camMat')

for n=100:100
    n
    camCordDisp=1;
    %Read Depth, Color and sgmentation mask
    base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
    depth=imread(sprintf('%stest/%02d/depth/%04d.png',base_path,objId,n));
    color=im2double(imread(sprintf('%stest/%02d/rgb/%04d.png',base_path,objId,n)));
    seg=imread(sprintf('%stest/%02d/seg/%04d.png',base_path,objId,n));
    gtlabelImg=imread(sprintf('%stest/%02d/64bins/labelled/%04d.png',base_path,objId,n));
    
    
    objCal = zeros(size(seg,1),size(seg,2),4);
    labelImg = zeros(size(seg),'uint8');
    objLabel =zeros(480,640,3);
    
    maxExtent= [0.100792;0.181796 ;0.193734 ];
    
    % Bins to save 3D Object coordinates falling in each bin
    bins3d=zeros(1,4);
    % Centrioids of 3D bins
    bin3DCents= zeros(power(binDim,3),3);
    
    % Bins to save the 2D pixels falling in each bin
    bins2d = zeros(1,3,'double');
    
    % Centroids of 2D bins
    bin2DCents= zeros(power(binDim,3),2);
    
    %3D projections of 2D centroids of bins
    cenProj3d = zeros(power(binDim,3),3);
    
    numPt=0;
    ptIdx=0 ;
    pc = zeros(307200,3);
    colorMap = zeros(307200,3);
    
    colors= distinguishable_colors(power(binDim,3),[0 0 0]);
    labels = generateLabels(binDim);
    %             [pc,colorMap] = reconstructScene(objId,n);
    
    %100
    r= [ 0.89526898  0.37082601 -0.246943  ;
        -0.113724   -0.34570301 -0.931427  ;
        -0.43076599  0.86196202 -0.26732501];
    t=[ -100.50654196;
        89.42205556;
        1038.79488726]/1000;
    
    
    toc = [r t;0 0 0 1];
    t0 = [0;0;0];
    toc1 = [rotx(180) t0;0 0 0 1];
    
    % Get the 3D boinding box
    bb3D = getBB3D(maxExtent);
    
    % If the object has to be displayed in camera coordinates, convert
    % bounding box in camera coordinates
    if(camCordDisp)
        bbHomo = [bb3D' ;ones(1,8)];
        bb3DCamCordH = (toc)*bbHomo;
        bb3DCamCord = bb3DCamCordH(1:3,:)./bb3DCamCordH(4,:);
        bb3D =bb3DCamCord';
    end
    
    % Find the min and max bounds of bounding box along all 3 dimensions
    minX = min(bb3D(:,1));
    maxX = max(bb3D(:,1));
    minY = min(bb3D(:,2));
    maxY = max(bb3D(:,2));
    minZ = min(bb3D(:,3));
    maxZ = max(bb3D(:,3));
    
    % Find the dimension of bin along each dimension
    unitX=(maxX-minX)/binDim;
    unitY = (maxY-minY)/binDim;
    unitZ= (maxZ-minZ)/binDim;
    
    % Now we use the detected object coordinate label image
    %Iterate through image, and for the pixels inside segmentation mask,
    %note  down the bin corresponding to each pixel
    for x=1:640
        for y=1:480
            ptIdx=ptIdx+1;
            if(seg(y,x) >0)
                d = depth(y,x);
                if( d~=0)
                    label =gtlabelImg(y,x);
                    numPt=numPt+1;
                    bins2d(numPt,1:3)=[double(x) double(y)  double(label)] ;
                    
                end
            end
        end
    end
    
    imshow(color);
    
    % Now we calculate the 2D centroids of bins in the label image.
    % We also find the 3D projections of the 2D bin centroids.
    hold on
    ct=1;
    binIds2D = unique(bins2d(:,3))
    for  b =1:size(binIds2D,1)
        
        id = binIds2D(b);
        if(id ~= 0)
            num = size (find(bins2d(:,3)==id),1);
            bx = sum(bins2d(find(bins2d(:,3)==id),1))/num;
            by = sum( bins2d( find(bins2d(:,3)==id),2))/num;
            bin2DCents(id,:)=[bx,by];
            centD = depth(round(by),round(bx));
            
        end
        ct=ct+1;
    end
    
    binPC1 = pointCloud(cenProj3d,'Color',colors);
    
    %     figure
    %     pcshow(binPC1,'MarkerSize',200,'VerticalAxis','Y', 'VerticalAxisDir','Up');
    numPt=0;
    for x=1:640
        for y=1:480
            ptIdx=ptIdx+1;
            if(seg(y,x) >0)
                d = depth(y,x);
                if( d~=0)
                    xc=(double(x)-325.2611)*(double(d)/(1000.0*572.4114));
                    yc=(y-242.04899)*(double(d)/(1000*573.57043));
                    zc=double(d)/1000;
                    
                    camC=[xc;yc;zc;1];
                    xo = (camC);
                    if(camCordDisp==0)
                        xo = toc1*(inv(toc)*xo);     % Obj Coord
                    else
                        xo = toc1*xo;
                    end
                    
                    objCal(y,x,1) = xo(1,1)+maxExtent(1)/2;
                    objCal(y,x,1) = (objCal(y,x,1)/maxExtent(1));
                    
                    objCal(y,x,2) = xo(2,1)+maxExtent(2)/2;
                    objCal(y,x,2) = (objCal(y,x,2)/maxExtent(2));
                    
                    objCal(y,x,3) = xo(3,1)+maxExtent(3)/2;
                    objCal(y,x,3) = (objCal(y,x,3)/maxExtent(3));
                    
                    obX = xo(1,1);
                    obY = xo(2,1);
                    obZ = xo(3,1);
                    if( (obX>minX) && (obX<maxX) && (obY > minY) && (obY<maxY) ...
                            && (obZ>minZ) && (obZ<maxZ))
                        numPt=numPt+1;
                        
                        lx=  ceil( (obX-minX)/unitX);
                        if(lx==0)
                            lx=1;
                        end;
                        ly=  ceil( (obY-minY)/unitY);
                        if(ly==0)
                            ly=1;
                        end;
                        lz=  ceil( (obZ-minZ)/unitZ);
                        if(lz==0)
                            lz=1;
                        end;
                        
                        strLabel = sprintf('%d%d%d',lx,ly,lz);
                        
                        l = find(labels== ( str2num(strLabel) ));
                        bins3d(numPt,1:4)=[xo(1,1) xo(2,1) xo(3,1) l] ;
                        
                        objLabel(y,x,1)=colors(l,1);
                        objLabel(y,x,2)=colors(l,2);
                        objLabel(y,x,3)=colors(l,3);
                        labelImg(y,x) = l;
                        
                        
                        pc(ptIdx,1) = xo(1,1);
                        pc(ptIdx,2) = xo(2,1);
                        pc(ptIdx,3) = xo(3,1);
                        
                        colorMap(ptIdx,1) = colors(l,1);
                        colorMap(ptIdx,2) = colors(l,2);
                        colorMap(ptIdx,3) = colors(l,3);
                        
                    else
                        pc(ptIdx,1) = xo(1,1);
                        pc(ptIdx,2) = xo(2,1);
                        pc(ptIdx,3) = xo(3,1);
                        
                        colorMap(ptIdx,1) = color(y,x,1);
                        colorMap(ptIdx,2) = color(y,x,2);
                        colorMap(ptIdx,3) = color(y,x,3);
                    end
                    
                end
            end
        end
        
    end
    
    %     figure
    %     imshow(objCal(:,:,1:3));
        figure
        imshow(objLabel);
    % %     figure
    % %     imshow(labelImg);
    %     %
    %     % % fnameLabel = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/objCordLabel/Can/label_%05d.png',n);
    %     % % imwrite(labelImg,fnameLabel);
    %     % %
    %     % % fnameColor = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/objCordColor/Can/cordColor_%05d.png',n);
    %     % % imwrite(objLabel,fnameColor);
    %     % % fclose('all');
    %     % %
    %
    orpc = pointCloud(pc,'Color',colorMap);
    
    
    
    binIds3D = unique(bins3d(:,4))
    for  b =1:size(binIds3D,1)
        id = binIds3D(b);
        if(id~=0)
            num = size (find(bins3d(:,4)==id),1);
            bx = sum(bins3d(find(bins3d(:,4)==id),1))/num;
            by = sum( bins3d( find(bins3d(:,4)==id),2))/num;
            bz = sum(bins3d(find(bins3d(:,4)==id),3))/num;
            bin3DCents(id,:)=[bx,by,bz];
        end
    end
    binPC = pointCloud(bin3DCents,'Color',colors);
    
    %     figure
    %     pcshow(binPC,'MarkerSize',200,'VerticalAxis','Y', 'VerticalAxisDir','Up');
    
%     [worldOrientation,worldLocation] = estimateWorldCameraPose(...
%         bin2DCents,bin3DCents,cam)
    
    figure
    pcshow(orpc,'VerticalAxis','Y', 'VerticalAxisDir','Up');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    hold on
    %
    
%     bb3D = getBB3D(maxExtent);
%     toc = [worldOrientation worldLocation';0 0 0 1];
%     
%     % If the object has to be displayed in camera coordinates, convert
%     % bounding box in camera coordinates
%     if(camCordDisp)
%         bbHomo = [bb3D' ;ones(1,8)];
%         bb3DCamCordH = (toc)*bbHomo;
%         bb3DCamCord = bb3DCamCordH(1:3,:)./bb3DCamCordH(4,:);
%         bb3D =bb3DCamCord';
%     end
    
    drawBB3D(bb3D)
    
end

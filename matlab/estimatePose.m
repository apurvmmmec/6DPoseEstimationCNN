% close all;
clear all;
objId=5;
binDim =2;

%CAmera Intrinsic Param Matrix
camMat = [572.41140   0          325.2611;
    0           573.57043  242.04899;
    0           0          1];
cam = cameraParameters('IntrinsicMatrix', camMat')

for n=30:30
    n
    camCordDisp=0;
    maxExtent= [0.100792;0.181796 ;0.193734 ];
    
    numPt=0;
    ptIdx=0 ;
    pc = zeros(307200,3);
    colorMap = zeros(307200,3);
    %Read Depth, Color and sgmentation mask
    base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
    depth=imread(sprintf('%stest/%02d/depth/%04d.png',base_path,objId,n));
    color=im2double(imread(sprintf('%stest/%02d/rgb/%04d.png',base_path,objId,n)));
    seg=imread(sprintf('%stest/%02d/seg/%04d.png',base_path,objId,n));
    
    gtlabelImg=imread(sprintf('%stest/%02d/8bins/labelled/%04d.png',base_path,objId,n));
    gtlabelColored=imread(sprintf('%stest/%02d/8bins/labelled_Colored/%04d.png',base_path,objId,n));
    
    imshow(gtlabelColored)
    objCal = zeros(size(seg,1),size(seg,2),4);
    labelImg = zeros(size(seg),'uint8');
    objLabel =zeros(480,640,3);
    colors= distinguishable_colors(power(binDim,3),[0,0,0]);
    labels = generateLabels(binDim);
    %             [pc,colorMap] = reconstructScene(objId,n);
    
    
    r=[ 0.97441399  0.18984599 -0.12032   ;
        -0.0544436  -0.320014   -0.94584697;
        -0.218069    0.92819703 -0.30149001];
    t=[  121.14949491;
        -113.72310242;
        1030.73827876]/1000;
    
    
    toc = [r t;0 0 0 1];
    t0 = [0;0;0];
    toc1 = [rotx(180) t0;0 0 0 1];
    
    % Get the 3D boinding box
    bb3D = getBB3D(maxExtent);
    
    % If the object has to be displayed in camera coordinates, convert
    %     % bounding box in camera coordinates
    if(camCordDisp)
        bbHomo = [bb3D' ;ones(1,8)];
        bb3DCamCordH = toc1*(toc)*bbHomo;
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
                    xo = (inv(toc)*xo);     % Obj Coord
                    
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
                        %                         bins3d(numPt,1:4)=[xo(1,1) xo(2,1) xo(3,1) l] ;
                        
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
                        
                    end
                    
                end
            end
        end
        
    end
    
    
    orpc = pointCloud(pc,'Color',colorMap);
    
    figure
    pcshow(orpc,'VerticalAxis','Y', 'VerticalAxisDir','Up');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(0,0)
    hold on
    
    drawBB3D(bb3D)
    figure
    imshow(objLabel)
    
end

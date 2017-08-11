% close all;
clear all;
objId=5
for n=30:30
    n
    %Read Depth, Color and sgmentation mask
        base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'

    depth=imread(sprintf('%stest/%02d/depth/%04d.png',base_path,objId,n));
    color=im2double(imread(sprintf('%stest/%02d/rgb/%04d.png',base_path,objId,n)));
    seg=imread(sprintf('%stest/%02d/seg/%04d.png',base_path,objId,n));
    eg = imread(sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/seg/Can/seg%05d.bmp',n));
    
     r=[ 0.97441399  0.18984599 -0.12032   ;
        -0.0544436  -0.320014   -0.94584697;
        -0.218069    0.92819703 -0.30149001];
    t=[  121.14949491;
        -113.72310242;
        1030.73827876]/1000;
        toc = [r t;0 0 0 1];

    objCal = zeros(size(seg,1),size(seg,2),4);
    labelImg = zeros(size(seg),'uint8');
    % maxExtent = [0.0775993; 0.0917691; 0.0758686]
    % maxExtent = [0.127633 ;0.117457; 0.0670107];
    % maxExtent = [0.129444; 0.0982834 ;0.109656];
    
    maxExtent= [0.181796 ;0.193734; 0.100792];
    objC=zeros(1,3);
    numPt=0;
%     pc = zeros(307200,3);
%     colorMap = zeros(307200,3);
    
    [pc,colorMap] = getScenePc(5,n);
    for x=1:640
        for y=1:480
            if(seg(y,x) >0)
                numPt=numPt+1;
                d = depth(y,x);
                xc=(double(x)-325.2611)*(double(d)/(1000.0*572.4114));
                yc=(y-242.04899)*(double(d)/(1000*573.57043));
                zc=double(d)/1000;
                
                camC=[xc;yc;zc;1];
                xo = (inv(toc)*camC);
                
                objCal(y,x,1) = xo(1,1);%+maxExtent(1)/2;
%                 objCal(y,x,1) = (objCal(y,x,1)/maxExtent(1));
                
                objCal(y,x,2) = xo(2,1);%+maxExtent(2)/2;
%                 objCal(y,x,2) = (objCal(y,x,2)/maxExtent(2));
                
                objCal(y,x,3) = xo(3,1);%+maxExtent(3)/2;
%                 objCal(y,x,3) = (objCal(y,x,3)/maxExtent(3));
                
                objC(numPt,1)=objCal(y,x,1);
                objC(numPt,2)=objCal(y,x,2);
                objC(numPt,3)=objCal(y,x,3);
                
%                 pc(numPt,1) = xo(1,1);
%                 pc(numPt,2) = xo(2,1);
%                 pc(numPt,3) = xo(3,1);
                
%                 colorMap(numPt,1) = color(y,x,1);
%                 colorMap(numPt,2) = color(y,x,2);
%                 colorMap(numPt,3) = color(y,x,3);
                
                
            end
        end
        
    end
    
    
    
    objLabel =zeros(480,620,3);
    
    %outlier Detection in Object coordinates
    
    % Find median of x coordinates
%     medX = median(objC(:,1));
%     
%     % Find median of y coordinates
%     medY = median(objC(:,2));
%     
%     % FInd median of z coordinates
%     medZ = median(objC(:,3));
    
    % Find outliers in X
    outIdX =find(abs(objC(:,1))>(maxExtent(3)/2));
    for idx=1:size(outIdX,1)
%         if(outIdX(idx)==1)
            objC(outIdX(idx),:) = [0,0,0];
%         else
%             objC(outIdX(idx),:) = objC(outIdX(idx)-1,:);
%         end
    end
    
    % Find outliers in Y
    outIdY =find(abs(objC(:,2))>(maxExtent(1)/2));
     for idx=1:size(outIdY,1)
%          if(outIdY(idx)==1)
            objC(outIdY(idx),:) = [0,0,0];
%          else
%             objC(outIdY(idx),:) = objC(outIdY(idx)-1,:);
%         end
     end
%     
    % Find outliers in Z
    outIdZ =find(abs(objC(:,3))>(maxExtent(2)/2));
     for idx=1:size(outIdZ,1)
%          if(outIdZ(idx)==1)
            objC(outIdZ(idx),:) = [0,0,0];
%         else
%             objC(outIdZ(idx),:) = objC(outIdZ(idx)-1,:);
%         end
     end
    
    colors= distinguishable_colors(126,[0 ,0,0]);
    labels = generateLabels(5);
    
    maxX= max(objC(:,1));
    minX = min(objC(:,1));
    
    maxY= max(objC(:,2));
    minY = min(objC(:,2));
    
    maxZ= max(objC(:,3));
    minZ = min(objC(:,3));
    
    unitX=(maxX-minX)/5;
    
    unitY = (maxY-minY)/5;
    
    unitZ= (maxZ-minZ)/5;
    
    
    
    numPt=0;
    ptIdx=0;
    for x=1:640
        for y=1:480
            ptIdx=ptIdx+1;
            if(seg(y,x) >0)
                numPt=numPt+1;
                
                obX = objC(numPt,1);
                obY = objC(numPt,2);
                obZ = objC(numPt,3);
                
               if ((obX > minX) & (obX < maxX) & (obY > minY) & (obY < maxY) & ...
                             (obZ > minZ) & (obZ < maxZ))
                
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
                
                
%                 l=lx*ly+25*(lz-1);
                strLabel = sprintf('%d%d%d',lx,ly,lz);
                
                l = find(labels== ( str2num(strLabel) ));
                
                objLabel(y,x,1)=colors(l,1);
                objLabel(y,x,2)=colors(l,2);
                objLabel(y,x,3)=colors(l,3);
                labelImg(y,x) = l;
                
                colorMap(ptIdx,1) = colors(l,1);
                colorMap(ptIdx,2) = colors(l,2);
                colorMap(ptIdx,3) = colors(l,3);
               end
                
            end
        end
        
    end
%     figure
%     imshow(objCal(:,:,1:3));
    figure
    imshow(objLabel);
%     figure
%     imshow(labelImg);
    
%     fnameLabel = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/objCordLabel/Can/label_%05d.png',n);
% imwrite(labelImg,fnameLabel);
% 
% fnameColor = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/objCordColor/Can/cordColor_%05d.png',n);
% imwrite(objLabel,fnameColor);
% fclose('all');
%     
%     figure
    orpc = pointCloud(pc,'Color',colorMap);
    
    figure
    pcshow(orpc);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    hold on
    
    v1=[minX minY minZ ]
    v2=[maxX minY minZ]
    v3 = [maxX maxY minZ]
    v4= [minX maxY minZ]
    
    v5=[minX minY maxZ ]
    v6=[maxX minY maxZ]
    v7 = [maxX maxY maxZ]
    v8= [minX maxY maxZ]
    
    v=[v2;v1];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    hold on
    v=[v2;v3];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    hold on
    v=[v3;v4];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    hold on
    v=[v1;v4];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    hold on
    v=[v1;v5];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    hold on
    v=[v2;v6];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    hold on
    v=[v3;v7];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    hold on
    v=[v4;v8];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    
    hold on
    v=[v5;v6];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    

    hold on
    v=[v6;v7];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    
    hold on
    v=[v7;v8];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
    
    hold on
    v=[v5;v8];
    plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
end



function [pc, colorMap] = reconstructScene(objId,imId)

base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/';
depth=imread(sprintf('%stest/%02d/depth/%04d.png',base_path,objId,imId));
color=im2double(imread(sprintf('%stest/%02d/rgb/%04d.png',base_path,objId,imId)));
% seg=imread(sprintf('%stest/%02d/seg/%04d.png',base_path,objId,imId));


numPt=0;
pc = zeros(307200,3);
colorMap = zeros(307200,3);
t0 = [0;0;0]
toc1 = [rotx(180) t0;0 0 0 1];
for x=1:640
    for y=1:480
        %                                         if(seg(y,x) ==0)
        
        
        numPt=numPt+1;
        d = depth(y,x);
        xc=(double(x)-325.2611)*(double(d)/(1000.0*572.4114));
        yc=-(y-242.04899)*(double(d)/(1000*573.57043));
        zc=-double(d)/1000;
        
        camC=[xc;yc;zc;1];
        
        xo = (camC);
        
        pc(numPt,1) = xo(1,1);
        pc(numPt,2) = xo(2,1);
        pc(numPt,3) = xo(3,1);
        colorMap(numPt,1) = color(y,x,1);
        colorMap(numPt,2) = color(y,x,2);
        colorMap(numPt,3) = color(y,x,3);
        %                                         end
    end
end














% close all;
% clear all;
% objId=5
% for n=100:100
%     n
%     %Read Depth, Color and sgmentation mask
%     base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
%     depth=imread(sprintf('%stest/%02d/depth/%04d.png',base_path,objId,n));
%     color=im2double(imread(sprintf('%stest/%02d/rgb/%04d.png',base_path,objId,n)));
%     seg=imread(sprintf('%stest/%02d/seg/%04d.png',base_path,objId,n));
%
%     r=[ 0.98310202 -0.0472039  -0.17686699;
%         -0.167686   -0.61976397 -0.76666403;
%         -0.0734263   0.78336698 -0.61720699];
%     t=[   55.44819932;
%         -81.89323636;
%         1025.71399065]/1000;
%
%     r= [ 0.89526898  0.37082601 -0.246943  ;
%         -0.113724   -0.34570301 -0.931427  ;
%         -0.43076599  0.86196202 -0.26732501];
%     t=[ -100.50654196;
%         89.42205556;
%         1038.79488726]/1000;
%
%
%     toc = [r t;0 0 0 1];
%
%     r= [ 0.89526898  0.37082601 -0.246943  ;
%         -0.113724   -0.34570301 -0.931427  ;
%         -0.43076599  0.86196202 -0.26732501];
%     t=[ -100.50654196;
%         89.42205556;
%         1038.79488726]/1000;
%
%
%     toc1 = [r t;0 0 0 1];
%
%     objCal = zeros(size(seg,1),size(seg,2),4);
%     labelImg = zeros(size(seg),'uint8');
%
%     maxExtent= [0.181796 ;0.193734; 0.100792];
%     hf = maxExtent/2;
%     maxExtX = (((toc))*[hf(1) ;0 ;0; 1])
%     minExtX = (((toc))*[-hf(1) ;0 ;0; 1])
%
%
%     maxExtY = (((toc))*[0 ;hf(2) ;0; 1])
%     minExtY = (((toc))*[0 ;-hf(2) ;0; 1])
%
%     maxExtZ = (((toc))*[0 ;0 ;hf(3); 1])
%     minExtZ = (((toc))*[0 ;0 ;-hf(3); 1])
%
%     maxExtentXformed = [maxExtX(1);maxExtY(2);maxExtZ(3)]
%     minExtentXformed = [minExtX(1);minExtY(2);minExtZ(3)]
%
%
%     objC=zeros(1,3);
%     numPt=0;
%     pc = zeros(307200,3);
%     colorMap = zeros(307200,3);
%
% %     [pc,colorMap] = reconstructScene(objId,n);
%     ptIdx=0;
%
%     for x=1:640
%         for y=1:480
%             ptIdx=ptIdx+1;
%
%             if(seg(y,x) >0)
%                 d = depth(y,x);
%                 if( d~=0)
%
%                     numPt=numPt+1;
%                     xc=(double(x)-325.2611)*(double(d)/(1000.0*572.4114));
%                     yc=(y-242.04899)*(double(d)/(1000*573.57043));
%                     zc=double(d)/1000;
%
%                     camC=[xc;yc;zc;1];
%                     xo = (camC);
% %                     xo = (inv(toc)*xo);
% %                     xo = ((toc)*xo);
%
%
%                     objCal(y,x,1) = xo(1,1);%+maxExtent(1)/2;
%                     %                     objCal(y,x,1) = (objCal(y,x,1)/maxExtent(1));
%
%                     objCal(y,x,2) = xo(2,1);%+maxExtent(2)/2;
%                     %                     objCal(y,x,2) = (objCal(y,x,2)/maxExtent(2));
%
%                     objCal(y,x,3) = xo(3,1);%+maxExtent(3)/2;
%                     %                     objCal(y,x,3) = (objCal(y,x,3)/maxExtent(3));
%
%                     objC(numPt,1)=objCal(y,x,1);
%                     objC(numPt,2)=objCal(y,x,2);
%                     objC(numPt,3)=objCal(y,x,3);
%
%                     pc(ptIdx,1) = xo(1,1);
%                     pc(ptIdx,2) = xo(2,1);
%                     pc(ptIdx,3) = xo(3,1);
%
%                     colorMap(ptIdx,1) = color(y,x,1);
%                     colorMap(ptIdx,2) = color(y,x,2);
%                     colorMap(ptIdx,3) = color(y,x,3);
%
%                 end
%
%
%
%
%             end
%         end
%
%     end
%
%
%
%     objLabel =zeros(480,640,3);
%
%
%     % Find outliers in X
% %     outIdX =find(abs(objC(:,1))>(maxExtentXformed(3)/2));
%         outIdX =find( (objC(:,1)>minExtentXformed(3)) & (objC(:,1)<maxExtentXformed(3)));
%
%     for idx=1:size(outIdX,1)
%         objC(outIdX(idx),:) = [0,0,0];
%     end
%
%     % %         Find outliers in Y
% %     outIdY =find(abs(objC(:,2))>(maxExtentXformed(1)/2));
%         outIdY =find( (objC(:,2)>minExtentXformed(1)) & (objC(:,2)<maxExtentXformed(1)));
%
%     for idx=1:size(outIdY,1)
%         objC(outIdY(idx),:) = [0,0,0];
%     end
%
%     % %     Find outliers in Z
% %     outIdZ =find(abs(objC(:,3))>(maxExtentXformed(2)/2));
%         outIdZ =find( (objC(:,3)>minExtentXformed(2)) & (objC(:,3)<maxExtentXformed(2)));
%
%     for idx=1:size(outIdZ,1)
%         objC(outIdZ(idx),:) = [0,0,0];
%     end
%
%     colors= distinguishable_colors(64,[0 0 0]);
%     labels = generateLabels();
%
%     maxX= max(objC(:,1));
%     minX = min(objC(:,1));
%
%     maxY= max(objC(:,2));
%     minY = min(objC(:,2));
%
%     maxZ= max(objC(:,3));
%     minZ = min(objC(:,3));
%
%     unitX=(maxX-minX)/4;
%
%     unitY = (maxY-minY)/4;
%
%     unitZ= (maxZ-minZ)/4;
%
%
%
%     numPt=0;
%     ptIdx=0;
%     for x=1:640
%         for y=1:480
%             ptIdx=ptIdx+1;
%             if(seg(y,x) >0)
%                 d = depth(y,x);
%                 if(d ~=0)
%
%                     numPt=numPt+1;
%
%                     %                 obX = objC(numPt,1);
%                     %                 obY = objC(numPt,2);
%                     %                 obZ = objC(numPt,3);
%
%                     obX = objCal(y,x,1);
%                     obY = objCal(y,x,2);
%                     obZ = objCal(y,x,3);
%                     if( (obX>minX) && (obX<maxX) && (obY > minY) && (obY<maxY) ...
%                             && (obZ>minZ) && (obZ<maxZ))
%                         lx=  ceil( (obX-minX)/unitX);
%                         if(lx==0)
%                             lx=1;
%                         end;
%                         ly=  ceil( (obY-minY)/unitY);
%                         if(ly==0)
%                             ly=1;
%                         end;
%                         lz=  ceil( (obZ-minZ)/unitZ);
%                         if(lz==0)
%                             lz=1;
%                         end;
%
%
%                         %                 l=lx*ly+25*(lz-1);
%                         strLabel = sprintf('%d%d%d',lx,ly,lz);
%
%                         l = find(labels== ( str2num(strLabel) ));
%
%                         objLabel(y,x,1)=colors(l,1);
%                         objLabel(y,x,2)=colors(l,2);
%                         objLabel(y,x,3)=colors(l,3);
%                         labelImg(y,x) = l;
% %
%                         colorMap(ptIdx,1) = colors(l,1);
%                         colorMap(ptIdx,2) = colors(l,2);
%                         colorMap(ptIdx,3) = colors(l,3);
%                     end
%
%                 end
%             end
%         end
%
%     end
%     figure
%     imshow(objCal(:,:,1:3));
%     figure
%     imshow(objLabel);
%     figure
%     imshow(labelImg);
%     %
%     % %     fnameLabel = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/objCordLabel/Can/label_%05d.png',n);
%     % % imwrite(labelImg,fnameLabel);
%     % %
%     % % fnameColor = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/objCordColor/Can/cordColor_%05d.png',n);
%     % % imwrite(objLabel,fnameColor);
%     % % fclose('all');
%     % %
%
%     orpc = pointCloud(pc,'Color',colorMap);
%
%     figure
%     pcshow(orpc);
%     xlabel('X');
%     ylabel('Y');
%     zlabel('Z');
%     hold on
%
% %     v1=[minX minY minZ ];
% %     v2=[maxX minY minZ];
% %     v3 = [maxX maxY minZ];
% %     v4= [minX maxY minZ];
% %
% %     v5=[minX minY maxZ ];
% %     v6=[maxX minY maxZ];
% %     v7 = [maxX maxY maxZ];
% %     v8= [minX maxY maxZ];
%
%     v1 = [minExtX(1) minExtY(2) minExtZ(3) ];
%     v2 = [maxExtX(1) minExtY(2) minExtZ(3)];
%
%     v3 = [maxExtX(1) maxExtY(2) minExtZ(3)];
%     v4 = [minExtX(1) maxExtY(2) minExtZ(3)];
%
%     v5 =[minExtX(1) minExtY(2) maxExtZ(3) ];
%     v6 = [maxExtX(1) minExtY(2) maxExtZ(3)];
%
%     v7 = [maxExtX(1) maxExtY(2) maxExtZ(3)];
%     v8= [minExtX(1) maxExtY(2) maxExtZ(3)];
%
%     v=[v2;v1];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%     hold on
%     v=[v2;v3];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%     hold on
%     v=[v3;v4];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%     hold on
%     v=[v1;v4];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%     hold on
%     v=[v1;v5];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%     hold on
%     v=[v2;v6];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%     hold on
%     v=[v3;v7];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%     hold on
%     v=[v4;v8];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%
%     hold on
%     v=[v5;v6];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%
%
%     hold on
%     v=[v6;v7];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%
%     hold on
%     v=[v7;v8];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
%
%     hold on
%     v=[v5;v8];
%     plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',3)
% end
%
%
%
%
%
%
%

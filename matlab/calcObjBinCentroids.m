function calcObjBinCentroids()
close all
clear all
objId=02
binDim=5
base_path = '/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/'
ptCloud = pcread(sprintf('%smodels/obj_02.ply',base_path));
% glassCloud = pcread(sprintf('%smodels/rayban.ply',base_path));

r=[ 0.29869699  0.862638   -0.40821001;
  -0.249641   0.48346901 0.83901   ;
 0.92111897  -0.14870299 0.35975999];
t=[  -13.46474647
   -70.33066627
  1038.43006007];

obj = ptCloud.Location;
% cl = ptCloud.Color;
% nl= ptCloud.Normal;
% glass = (glassCloud.Location/14)*rotz(-90)*roty(10);
% num3DPtsGlass= size(glass,1);
% 
% 
% glassColor = glassCloud.Color;
% glass(:,1)=glass(:,1)/1.2;
% glass(:,1)=glass(:,1)+13.0;
% 
% glass(:,2)=glass(:,2)*1.2;
% glass(:,3)=glass(:,3)+17.0;

% cent = sum(glass)/num3DPtsGlass;
% glass = glass - cent;
% for i =1:num3DPtsGlass
%     if (glass(i,2)<0)
%         glass(i,:)=glass(i,:)*rotz(20);
%     else
%         glass(i,:)=glass(i,:)*rotz(-20);
% 
%     end
% end
% glass= glass+cent;

% pcwrite(pointCloud(glass),'glass.ply')
% glass=glass*rotx(90);

% obj=obj*rotz(180)*rotx(180)/1000;
obj=obj;

pc = zeros(307200,3);
colorMap = zeros(307200,3);
% nlMap = zeros(307200,3);
% glassPC = zeros(num3DPtsGlass,3);
% glassColorMap = zeros(num3DPtsGlass,3);



% maxExtent= [0.100792;0.181796 ;0.193734 ]*1000; %can

% maxExtent = [67.01070000;127.63300000 ;117.45660000] %cat


% maxExtent=[0.0758686;0.0775993;0.0917691; ]*1000 %Ape
% maxExtent = [104.42920000 ;77.40760000 ; 85.69700000]; %Duck

% maxExtent =[0.13665940000; 0.14303020000; 0.10049700000]*1000 ; % Camera

% maxExtent =[203.14600000; 117.75250000;213.11600000] %Lamp
% maxExtent = [258.2260 ; 118.4821 ; 141.1324] % Iron
maxExtent = [215.67000000; 121.85570000;219.41000000] % Benchvise
num3DPts= size(obj,1)
colors= distinguishable_colors(power(binDim,3),[0,0,0]);
labels = generateLabels(binDim);
bin3DCents= zeros(power(binDim,3),4);

% Get the 3D boinding box
bb3D = getBB3D(maxExtent);


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

numPt=0

for i=1:num3DPts
    i;
    obX = obj(i,1);
    obY = obj(i,2);
    obZ = obj(i,3);
    
    numPt=numPt+1;
    
    lx=  ceil( (obX-minX)/unitX);
    if(lx==0)
        lx=1;
    elseif(lx >binDim)
        lx=binDim
    end;
    ly=  ceil( (obY-minY)/unitY);
    if(ly==0)
        ly=1;
    elseif(ly >binDim)
        ly=binDim
    end;
    lz=  ceil( (obZ-minZ)/unitZ);
    if(lz==0)
        lz=1;
    elseif(lz >binDim)
        lz=binDim
    end;
    
    strLabel = sprintf('%d%d%d',lx,ly,lz);
    
    l = find(labels== ( str2num(strLabel) ));
    bins3d(numPt,1:4)=[obX  obY obZ l] ;
    
    
    pc(i,1) = obj(i,1);
    pc(i,2) = obj(i,2);
    pc(i,3) = obj(i,3);
    
    colorMap(i,1) = colors(l,1);
    colorMap(i,2) = colors(l,2);
    colorMap(i,3) = colors(l,3);
    
%      colorMap(i,1) = double(cl(i,1))/255.0;
%     colorMap(i,2) = double(cl(i,2))/255.0;
%     colorMap(i,3) = double(cl(i,3))/255.0;
    
%     nlMap(i,1) = nl(i,1);
%     nlMap(i,2) = nl(i,2);
%     nlMap(i,3) = nl(i,3);
    
end


% for i=1:num3DPtsGlass
%     
%     
%     glassPC(i,1) = glass(i,1);
%     glassPC(i,2) = glass(i,2);
%     glassPC(i,3) = glass(i,3);
%     
%     glassColorMap(i,1) = 0;
%     glassColorMap(i,2) = 0;
%     glassColorMap(i,3) = 0;
%     
% end
orpc = pointCloud(pc,'Color',colorMap);
% orpc = pointCloud(pc);

pcshow(orpc)
xlabel('X');
ylabel('Y');
zlabel('Z');
% view(0,0)

hold on
% 
% gPc = pointCloud(glassPC,'Color',glassColorMap);
% pcshow(gPc)
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% % view(0,0)
% 
% hold on

drawBB3D(bb3D)

binIds3D = unique(bins3d(:,4))
for  b =1:size(binIds3D,1)
    id = binIds3D(b);
    if(id~=0)
        num = size (find(bins3d(:,4)==id),1);
        bx = sum(bins3d(find(bins3d(:,4)==id),1))/num;
        by = sum( bins3d( find(bins3d(:,4)==id),2))/num;
        bz = sum(bins3d(find(bins3d(:,4)==id),3))/num;
        %             if((id ==103)|(id ==123)|(id ==81)|(id ==12))
        bin3DCents(id,:)=[bx,by,bz, id];
        %             end
    end
end
binPC = pointCloud(bin3DCents(:,1:3),'Color',colors);

% 
% pcshow(binPC,'MarkerSize',800,'VerticalAxis','Y', 'VerticalAxisDir','Up');
% view(0,0)
zoom(0.7)
% for i=30:1:330 
%      
% view(i,30)
% pause(.01)
%     
% end
for i=1:125
    disp(sprintf('%f,  %f,  %f,',bin3DCents(i,1),bin3DCents(i,2),bin3DCents(i,3)))
end
end
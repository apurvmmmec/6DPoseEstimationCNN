clear all;
close all;
addpath(genpath('/Users/apurvnigam/study_ucl/term1/MScThesis/code/autoContextRF/dummy_data/YAMLMatlab_0.4.3'));

yaml_file = '../../../hinterstoisser/train/06/gt.yml';
YamlStruct = ReadYaml(yaml_file);

for i =0:0
        cImgName=sprintf('../../../hinterstoisser/train/06/rgb/%04d.png',i);
     cImg = imread(cImgName);
    dImg = imread(sprintf('../../../hinterstoisser/train/06/depth/%04d.png',i));
    mask = imread(sprintf('../../../hinterstoisser/train/06/seg/%04d.png',i));
    objCal = zeros(size(mask,1),size(mask,2),4);
    %
    r= [1.00000000, 0.00000000, 0.00000000;
        0.00000000, -1.00000000, -0.00000000;
        0.00000000, 0.00000000, -1.00000000];
    t= [-0.00000000; 0.00000000; 0.40000000];
    toc = [r t;0 0 0 1];
    %
    maxExtent = [0.127633 0.117457 0.0670107];
    
    % gray = rgb2gray(cImg);
    % imshow(dImg(:,:,3),[]);
    % mask = gray > 0;
    % figure
    % imshow(seg);
    % maskName= sprintf('../../../hinterstoisserDataset/train/06/seg/%04d.png',i)
    % imwrite(mask,maskName);
    objC=zeros(1,3);
    ct=0;
    for x=1:640
        for y=1:480
            
            if(mask(y,x) >0)
                ct=ct+1;
                d = dImg(y,x);
                xc=(double(x)-320.0)*(double(d)/(1000.0*575.816));
                yc=-(y-240)*(double(d)/(1000*575.816));
                zc=-double(d)/1000;
                
                camC=[xc;yc;zc;1];
                xo = ((toc)*camC);
                objC(ct,1)=xo(1,1);
                objC(ct,2)=xo(2,1);
                objC(ct,3)=xo(3,1);
                
                objCal(y,x,1) = xo(1,1);%+maxExtent(3)/2;
                %                 objCal(y,x,3) = (objCal(y,x,3)/maxExtent(3));
                
                objCal(y,x,2) = xo(2,1);%+maxExtent(2)/2;
                %                 objCal(y,x,2) = (objCal(y,x,2)/maxExtent(2));
                
                objCal(y,x,3) = xo(3,1);%+maxExtent(1)/2;
                %                 objCal(y,x,1) = (objCal(y,x,1)/maxExtent(1));
                
                
                
            end
        end
        
    end
    
    maxX= max(objC(:,1));
    minX = min(objC(:,1));
    maxY= max(objC(:,2));
    minY = min(objC(:,2));
    maxZ= max(objC(:,3));
    minZ = min(objC(:,3));
    unitX=(maxX-minX)/5;
    unitY = (maxY-minY)/5;
    unitZ= (maxZ-minZ)/5;
    for x=1:640
        for y=1:480
            
            if(mask(y,x) >0)
                
                obX = objCal(y,x,1);
                obY = objCal(y,x,2);
                obZ = objCal(y,x,3);
                
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
                
                
                l=lx*ly+25*(lz-1);
                objCal(y,x,4)=l;
                
                
                
                
                
                
            end
        end
        
    end
        imshow(objCal(:,:,4),[]);

    
end
close all

fid = fopen('../../../hinterstoisser/OcclusionChallengeICCV2015/models/Can/004.xyz');
tline = fgetl(fid);
obj = zeros(1,3);
i=0;
while ischar(tline)
    i=i+1;
    C = strsplit(tline,' ');
    
    obj(i,1) = str2double(C(1))/1;
    obj(i,2) = str2double(C(2))/1;
    obj(i,3) = str2double(C(3))/1;
    
    tline = fgetl(fid);
end
fclose(fid);

numPts = i;

for n=100:100
    
    poseMat=getTransf(n);
    img = zeros(480,640,'uint8');
    
    for idx=1:numPts
        x = obj(idx,1);
        y = obj(idx,2);
        z = obj(idx,3);
        
        camCord = (poseMat)*[x;y;z;1];
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
            end
    
%     fname = sprintf('../../../hinterstoisser/OcclusionChallengeICCV2015/seg/Can/seg%05d.bmp',n);
    img=im2double(img);
    imshow(img)
%     imwrite(img,fname);
    fclose('all')
    
end



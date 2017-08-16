function loadModel()

model_path= '/Users/apurvnigam/study_ucl/term1/MScThesis/dataset/can/object.xyz'
fid = fopen(model_path);
tline = fgetl(fid);
obj = zeros(1,3);
i=0;
while ischar(tline)
    if(i==0)
        tline = fgetl(fid);
        i=i+1;
    else
        C = strsplit(tline,' ');
        obj(i,1) = str2double(C(1))*1;
        obj(i,2) = str2double(C(2))*1;
        obj(i,3) = str2double(C(3))*1;
        tline = fgetl(fid);
        i=i+1;
    end
    
end
fclose(fid);
ptCloud = pointCloud(obj);
figure
pcshow(ptCloud);
xlabel('X');
ylabel('Y');
zlabel('Z');




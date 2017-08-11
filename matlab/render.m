
close all
clear all
img = zeros(480,640);
ptCloud = pcread('test/Accv_Cat/mesh.ply');
pcshow(ptCloud);

xlabel('X');
ylabel('Y');
zlabel('Z');


camMat = [572.41140         0 325.2611  0;
    0 573.57043 242.04899 0;
    0         0         1 0];


r=[-0.49824 -0.37171 0.783422;
0.648946 0.439464 0.621212;
-0.575142 0.817844 0.0222681];

r=[-0.418426257384 -0.415351674981 0.807796584838;
0.672933128829 0.455603128078 0.582855394736;
-0.610091515428 0.787410728409 0.0888552427139];

t= [0.438188692957; 0.245032980654; -1.0839189668
];

toc=[r t;0 0 0 1];


fid = fopen('pc2.xyz');
tline = fgetl(fid);
numPts=0;
i=0;
while ischar(tline)
    i=i+1;
    C = strsplit(tline,' ');
    
    if(i==1)
        numPts= C(1);
        pc = zeros(str2double(C(1)),3);
    else
        x = str2double(C(1))/100;
        y = str2double(C(2))/100;
        z = str2double(C(3))/100;
        %         [rt]= ry'*[x;y;z;1];
        pc(i-1,1)=-y;
        pc(i-1,2)=z;
        pc(i-1,3)=x;
        %         pt = camMat*(toc)*[-y;-z;x;1];
        
        camCord = (toc)*[-y;-z;x;1];
        xc = camCord(1,1);
        yc = camCord(2,1);
        zc = camCord(3,1);
        d  = -zc;
        ptx = ((xc*(575.816)/d)+320);
        pty = (-(yc*(575.816)/d)+240);
        
        if((pty<480) && (pty >=1)) && ((ptx<640) &&(ptx >=1))    
            img(round(pty),round(ptx))=1;
        end
    end
    
    tline = fgetl(fid);
end

fclose(fid);
figure;
orpc = pointCloud(pc);
pcshow(orpc);
xlabel('X');
ylabel('Y');
zlabel('Z');

pcshow(orpc,'VerticalAxisDir','Down','MarkerSize',3000);
figure
% end
imshow(img);



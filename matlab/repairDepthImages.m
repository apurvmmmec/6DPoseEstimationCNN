close all
clear all

objId=4
h=480
w=640
for n=0:1200
    n
    %Read Depth, Color and sgmentation mask
    dImgPath = sprintf('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/%02d/depth/%04d.png',objId,n);
    segImgPath = sprintf('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/%02d/seg/%04d.png',objId,n);
    correctedDepthPath = sprintf('/Users/apurvnigam/study_ucl/term1/MScThesis/hinterstoisser/test/%02d/depth/%04d.png',objId,n);

    depth=imread(dImgPath);
    seg=imread(segImgPath);
%     figure
%     imshow(depth,[])
    for x=2:w-1
        for y=2:h-1
            if(seg(y,x) >0)
                d = depth(y,x);
                if (d==0)
                    dUp = (depth(y-1,x)>0);
                    dLeft = depth(y,x-1)>0;
                    dRight = depth(y,x+1)>0;
                    dDown =depth(y+1,x)>0;
                    dc = 0;
                    if ((seg(y-1,x)>0) && dUp)
                        d=d+depth(y-1,x);
                        dc=dc+1;
                    end
                    if ((seg(y+1,x)>0) && dDown)
                        d=d+depth(y+1,x);
                        dc=dc+1;
                        
                    end
                    if ((seg(y,x-1)>0) && dLeft)
                        d=d+depth(y,x-1);
                        dc=dc+1;
                        
                    end
                    if ((seg(y,x+1)>0) && dRight)
                        d=d+depth(y,x+1);
                        dc=dc+1;
                        
                    end
                    d=d/dc;
                    depth(y,x)=d;
                end
            end
        end
    end
%     figure
%     imshow(depth,[])
    imwrite((depth),correctedDepthPath);
end
    
    
close all
rows=480;
cols=640;
w=50;
h=50;
height=100;
width=100;
cenR=120
cenC=345
img=ones(rows,cols);
img(cenR-h:cenR+h,cenC-w:cenC+w) = 0.5;
figure
imshow(img)
hold on
plot(cenC,cenR,'g*');

for r =(0:3:rows - height)
    for c =(0:3:cols-width)
        patchCenR = r + h;
        patchCenC = c + w;
%         plot(patchCenC,patchCenR,'bx')
        %             label = int(labelImg[r + h - 1, c + w - 1])
        %             rgbSubImg = rgbImg[r:r + height - 1, c:c + width - 1];
        %             depSubImg = depImg[r:r + height - 1, c:c + width - 1];
        
        
        if ( (patchCenR >= (cenR - h)) && ...
                (patchCenR <= (cenR + h)) && ...
                (patchCenC >= (cenC - w)) && ...
                (patchCenC <= (cenC + w)))
            
            if((rem(r,10)==0) && (rem(c,10)==0))
                plot(patchCenC,patchCenR,'rx')

%                 plot(c,r,'rx')
            end
        else
            if((rem(r,120)==0) && (rem(c,120)==0))
                plot(c,r,'rx') 
            end
        end
    end
end
    

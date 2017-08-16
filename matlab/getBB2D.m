[pc,colorMap] = reconstructScene(2,100);


orpc = pointCloud(pc);
    figure
    pcshow(orpc);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(0,0)
    hold on
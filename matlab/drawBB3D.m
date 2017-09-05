function drawBB(bb3D)

v1 = bb3D(1,1:3);
v2 = bb3D(2,1:3);
v3 = bb3D(3,1:3);
v4 = bb3D(4,1:3);
v5 = bb3D(5,1:3);
v6 = bb3D(6,1:3);
v7 = bb3D(7,1:3);
v8 = bb3D(8,1:3);
v=[v2;v1];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
hold on
v=[v2;v3];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
hold on
v=[v3;v4];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
hold on
v=[v1;v4];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
hold on
v=[v1;v5];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
hold on
v=[v2;v6];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
hold on
v=[v3;v7];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
hold on
v=[v4;v8];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)

hold on
v=[v5;v6];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)


hold on
v=[v6;v7];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)

hold on
v=[v7;v8];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)

hold on
v=[v5;v8];
plot3(v(:,1),v(:,2),v(:,3),'r','LineWidth',1)
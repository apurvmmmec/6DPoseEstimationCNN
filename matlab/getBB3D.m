function bbObjCord = getBB3D(maxExtent)

bbminX = -maxExtent(1)/2;
bbmaxX =  maxExtent(1)/2;

bbminY = -maxExtent(2)/2;
bbmaxY =  maxExtent(2)/2;

bbminZ = -maxExtent(3)/2;
bbmaxZ =  maxExtent(3)/2;

v1 = [ bbminX bbminY bbminZ ];
v2 = [ bbmaxX bbminY bbminZ ];
v3 = [ bbmaxX bbmaxY bbminZ ];
v4 = [ bbminX bbmaxY bbminZ ];

v5 = [ bbminX bbminY bbmaxZ ];
v6 = [ bbmaxX bbminY bbmaxZ ];
v7 = [ bbmaxX bbmaxY bbmaxZ ];
v8 = [ bbminX bbmaxY bbmaxZ ];

bbObjCord = [v1; v2 ;v3; v4 ;v5; v6 ;v7 ;v8];

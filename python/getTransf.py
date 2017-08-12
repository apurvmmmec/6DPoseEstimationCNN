import numpy as np

def getTransf(num):
    fname = './poses/Can/info_%05d.txt'%(num);
    # file = open(fname, "r")
    # print(file)

    # tline = fgetl(fid);
    i = 0;
    r = np.zeros((3,3));
    t = np.zeros((3, 1));
    linesRead=0;
    # while ischar(tline)
    with open(fname) as file:
        for line in file:
            i = i + 1;
            C = line.split(" ")

            if (i == 5):
                # print(C)

                r[0,0] = float(C[0])
                r[0,1] = float(C[1])
                r[0,2] = float(C[2])
                linesRead=linesRead+1;
            elif (i == 6):
                # print(C)

                r[1, 0] = float(C[0])
                r[1, 1] = float(C[1])
                r[1, 2] = float(C[2])
                linesRead=linesRead+1;

            elif (i == 7):
                # print(C)

                r[2, 0] = float(C[0])
                r[2, 1] = float(C[1])
                r[2, 2] = float(C[2])
                linesRead=linesRead+1;

            elif (i ==9):
                # print(C)

                t[0, 0] = float(C[0]);
                t[1, 0] = float(C[1]);
                t[2, 0] = float(C[2]);
                linesRead=linesRead+1;



            xform = np.hstack((r ,  t))
            xform = np.vstack((xform,[0,0,0,1]))

    return xform,linesRead
# print(xform)
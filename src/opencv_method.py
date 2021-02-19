import numpy as np
from matplotlib import pylab as plt
import math
from collections import OrderedDict
import cv2



filename = 'Data/comb.img'
output = 'image4test/outputCV.img'
output1 = 'image4test/BTimg.img'

with open(filename , 'rb') as in_file:
    with open(output, 'wb') as out_file:
        out_file.write(in_file.read()[512:])

fo = open(output, 'rb')

Output512 = np.fromfile(output, dtype='uint8', sep="")
Output512 = Output512.reshape([512, 512])

with open(output , 'rb') as out_file1:
    myArr = bytearray(out_file1.read())


# Threshold the image for the value T = 128 to generate the binary image Bt
i = 0
myArrNew = myArr
for value in myArr :
    if(value > 128 ) :
        myArrNew[i] = 0
    else:
        myArrNew[i] = 255
    i = i + 1
    
print(512*512)
print(i)
with open(output1, 'wb') as out_file:
    out_file.write(myArrNew)

BTimage = np.fromfile(output1, dtype='uint8', sep="")
BTimage = BTimage.reshape([512, 512])
# cv2.imshow('image',BTimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def CCL(BTimage, sizeFilterValue): 
    # labels = []
    # stats = []
    # centroids = []
    # connectivity = 0
    # ltype = 0
    # ccltype = 0 
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(BTimage, 8)
    print("retval  = ")
    print(retval)

    print("labels  = ")
    print(labels)

    print("stats  = ")
    print(stats)

    print("centroids  = ")
    print(centroids)
    cv2.imwrite("image4test/labels.png",np.array(labels))


CCL(BTimage,7000)
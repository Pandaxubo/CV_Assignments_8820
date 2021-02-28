import numpy as np
from matplotlib import pyplot as plt
import math
from collections import OrderedDict

#build Distance Transfrom for generation of Medial Axis#
#Assignment 2#
filename = 'input-images/comb.img'
output = 'input-images/outputCV.img'
output1 = 'input-images/BTimg.img'

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
    if(value <= 128 ) :
        myArrNew[i] = 2 #border
    else:
        myArrNew[i] = 0 #component
    i = i + 1

with open(output1, 'wb') as out_file:
    out_file.write(myArrNew)

BTimage = np.fromfile(output1, dtype='uint8', sep="")
BTimage = BTimage.reshape([512, 512])


def buildDistanceTransfrom(BTimage):
    #set all values of pixels to 1 which are not 0
    for x in range(0,512,1):
        for y in range(0,512,1):
            if BTimage[x][y] != 0:
                BTimage[x][y] = 1

    counter = 0
    while True:
        try:
            BTimageold = BTimage
            tempImage1 = np.copy(BTimageold)
            counter = counter + 1
            for x in range(0,512,1):
                for y in range(0,512,1):
                    if BTimage[x][y] != 0:
                        if x - 1 >= 0 and y - 1 >= 0 and x + 1 < 512 and y + 1 < 512:
                            fourNeighbor = [int(BTimage[x][y]), int(BTimage[x - 1][y]), int(BTimage[x][y + 1]),
                                            int(BTimage[x + 1][y]), int(BTimage[x][y - 1])]
                            BTimage[x][y] = 1 + min(fourNeighbor)

            if np.array_equal(tempImage1,BTimage): #   tempImage1 is same as previous BTimage:
                break
        except:
            if np.array_equal(tempImage1, BTimage):
                break


    # generate medial axis
def generateMedialAxis(BTimage):
    tempImage = BTimage
    tempImageNew = np.copy(tempImage)
    medialAxisArray = np.copy(tempImage)
    for x in range(0,512,1):
        for y in range(0,512,1):
            tempImageNew[x][y] = 0
            medialAxisArray[x][y] = 0

    for x in range(0,512,1):
        for y in range(0,512,1):

            if tempImage[x][y] != 0 :
                if x - 1 >= 0 and y - 1 >= 0 and x + 1 < 512 and y + 1 < 512:


                    localmaxima = max(int(BTimage[x][y]),int(BTimage[x-1][y]),int(BTimage[x][y+1]),int(BTimage[x+1][y]),int(BTimage[x][y-1]))

                    if int(BTimage[x][y]) >= localmaxima:
                        tempImageNew[x][y] = 255
                        medialAxisArray[x][y] = BTimage[x][y]

    for x in range(0,512,1):
        for y in range(0,512,1):
            if BTimage[x][y] != 0 :
                BTimage[x][y] = 255
    plt.imshow(tempImageNew,cmap='gray')
    plt.show()

    return medialAxisArray



buildDistanceTransfrom(BTimage)


medialAxisArr = generateMedialAxis(BTimage)

def reconstructImageNew(medialAxisArr):
    # do post processing on the medial Axis Array

    ravelMedialArr = set(medialAxisArr.ravel())

    valueList = list(ravelMedialArr)  # contains all the unique lables in medial arr
    valueList.remove(0)

    tempArr = np.copy(medialAxisArr)    #copy the medial axis array to temperory array for processing


    while True:
        maximum = max(valueList)    # select maximum from the pixel set

        for x in range(0, 512, 1):
            for y in range(0, 512, 1):
                if tempArr[x][y] == maximum:

                    # extend left
                    i = x
                    j = y
                    leftdistance = int(tempArr[x][y])
                    tempArr[i][j] = tempArr[x][y]
                    while True:
                        j = j - 1
                        leftdistance = leftdistance - 1

                        if leftdistance <= 0 or j < 0:
                            break

                        if j >= 0:
                            if tempArr[i][j] < tempArr[i][j + 1]  :
                                tempArr[i][j] = tempArr[i][j + 1] - 1
                            else:
                                break

                    # extend right
                    i = x
                    j = y
                    rightdistance = int(tempArr[x][y])
                    tempArr[i][j] = tempArr[x][y]
                    while True:
                        j = j + 1
                        rightdistance = rightdistance - 1

                        if rightdistance <= 0 or j > 511:
                            break

                        if j <= 511:
                            if tempArr[i][j] < tempArr[i][j - 1] :
                                tempArr[i][j] = tempArr[i][j - 1] - 1
                            else:
                                break
                    # extend up
                    i = x
                    j = y
                    updistance = int(tempArr[x][y])
                    tempArr[i][j] = tempArr[x][y]
                    while True:
                        i = i - 1

                        updistance = updistance - 1

                        if updistance <= 0 or i < 0:
                            break

                        if i >= 0:
                            if tempArr[i][j] < tempArr[i + 1][j] :
                                tempArr[i][j] = tempArr[i + 1][j] - 1
                            else:
                                break
                    #extend down
                    i = x
                    j = y
                    downdistance = int(tempArr[x][y])
                    tempArr[i][j] = tempArr[x][y]
                    while True:
                        i = i + 1

                        downdistance = downdistance - 1

                        if downdistance <= 0 or i > 511:
                            break

                        if i <= 511:
                            if tempArr[i][j] < tempArr[i - 1][j]:
                                tempArr[i][j] = tempArr[i - 1][j] - 1
                            else:
                                break

        valueList.remove(maximum)       # delete one the maximum pixel value data is processed for the iteration

        if len(valueList) == 0:
            break

    # create binary image for display
    for x in range(0,512,1):
        for y in range(0,512,1):
            if tempArr[x][y] != 0 :
                tempArr[x][y] = 255

    plt.imshow(tempArr, cmap='gray')
    plt.show()


reconstructImageNew(medialAxisArr)
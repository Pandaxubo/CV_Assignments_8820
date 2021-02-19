import numpy as np
from matplotlib import pylab as plt
import math
from collections import OrderedDict

filename = 'Data/comb.img'
output = 'output/outputCV.img'
output1 = 'output/BTimg.img'

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
        myArrNew[i] = 2
    else:
        myArrNew[i] = 0
    i = i + 1

with open(output1, 'wb') as out_file:
    out_file.write(myArrNew)

BTimage = np.fromfile(output1, dtype='uint8', sep="")
BTimage = BTimage.reshape([512, 512])

#cv2.imwrite('output/thresd1.jpg', BTimage)


####    Iterative Conncected Component Labelling Algorithm Definition

def IterativeCCL(BTimage, sizeFilterValue):
    label = 5   # initialization of the label varaible
    i = 0
    j = 1
    equivalenceList = [[5]]

    tempImage = np.array(BTimage,dtype=int)

    for x in range(0,512,1):
        for y in range(0,512,1):
            if tempImage[x][y] == 2:
                if tempImage[x - 1][y] != tempImage[x][y - 1] and tempImage[x][y - 1] > 0 and tempImage[x - 1][y] > 0 and (x-1 >= 0 and y-1 >= 0):
                    if tempImage[x - 1][y] > tempImage[x][y - 1]:
                        tempImage[x][y] = tempImage[x][y - 1]
                        #logic using for loop to append the label
                        for item in equivalenceList:
                            for subItem in item:
                                if subItem == tempImage[x-1][y] and tempImage[x][y-1]!= 2:
                                    if tempImage[x][y-1] in item:
                                        break #pass
                                    else:
                                        item.append(tempImage[x][y - 1])
                                        break
                                pass  # pass for inner loop if
                            pass  # pass for outer loop if
                    else:
                        tempImage[x][y] = tempImage[x - 1][y]
                        for item in equivalenceList:
                            for subItem in item:
                                if subItem == tempImage[x - 1][y] and tempImage[x][y - 1] != 2:
                                    if tempImage[x][y-1] in item:
                                        break
                                    else:
                                        item.append(tempImage[x][y - 1])
                                        break
                                pass    # pass for inner loop else
                            pass    # pass for outer loop else
                else:
                    if tempImage[x - 1][y] > 0 and x - 1 >= 0:
                        tempImage[x][y] = tempImage[x - 1][y]
                    else:
                        if tempImage[x][y - 1] > 0 and y - 1 >= 0:
                            tempImage[x][y] = tempImage[x][y - 1]
                        else:
                            tempImage[x][y] = label
                            label = label + 1
                            if label == 286:
                                print('286')
                            j = 0
                            equivalenceList.append([label])
                            i = i + 1

    BTimage = tempImage

    uniqueEquivList = [list(OrderedDict.fromkeys(I)) for I in equivalenceList]

    newEQUIlist = equivalenceList

    #remove single element from the list
    for x in newEQUIlist:
        if len(x) == 1 :
            newEQUIlist.remove(x)

    #change labelling for the least in the sub-list
    CCLArraytemp = tempImage

    for x in range(0, 512, 1):
        for y in range(0, 512, 1):
            value1 = tempImage[x][y]
            if value1 > 2 :
                for subList in newEQUIlist:
                    if tempImage[x][y] in subList:
                        tempImage[x][y] = min(subList)

    valueset = set(tempImage.reshape(512 * 512))

    valueList = list(valueset)  # contains all the unique equivalence lables

    #create dictionary for equivalence table
    # the created dictionary is initialized with count to zero
    equivalenceDict = {}
    for x in range(0,len(valueList),1):
        equivalenceDict.setdefault(valueList[x], []).append(0)

    #display count of number of pixels for each label

    for x in range(0, 512, 1):
        for y in range(0, 512, 1):
            if tempImage[x][y] in valueList and tempImage[x][y] != 0:
                count = equivalenceDict.get(tempImage[x][y]) # return the value for the key from the dictionary
                increment = count[0]
                increment = increment + 1
                count[0] = increment
                equivalenceDict[tempImage[x][y]] = count

    ### Display Total number of components matching the size filter criteria ####

    graphicDisplayListImageC = []       # this list will contain the labels from equivalence table matching the size
                                        # filter criteria
    for key, value in equivalenceDict.items():
        if value[0] >= sizeFilterValue:
            graphicDisplayListImageC.append(key)

    # remove background count from the graphicDisplayDictImageC

    # Total number of components matching the criteria would be the items in the list graphicDisplayListImageC

    print('Total Number of Components matching the size filter criteria : ' , len(graphicDisplayListImageC))

    #### Assign unique gray level value to components which passes size filter criteria and all else background ####
    #### will be of pixel value 0 ####

    print('list of filterted components labels : ', graphicDisplayListImageC)

    basePixelVal = 0       # this value will be assigned to first filtered component. Later it will incremented by 20
                            #  to maintain the contrast in the image

    #build a dict for to determine new gray level value for the filtered components

    graphicDisplayDictImageC = {}

    for x in range(0,len(graphicDisplayListImageC),1):
        basePixelVal = basePixelVal + 30
        graphicDisplayDictImageC.setdefault(graphicDisplayListImageC[x], []).append(basePixelVal)


    # lookup in graphicDisplayDictImageC for the pixel value for the filtered component

    for x in range(0, 512, 1):
        for y in range(0, 512, 1):
            if tempImage[x][y] not in graphicDisplayListImageC:
                tempImage[x][y] = 0
            else:
                if tempImage[x][y] != 0:
                    newPixelVal = graphicDisplayDictImageC.get(tempImage[x][y])  # return the value for the key from the dictionary
                    tempnewPixelVal = newPixelVal[0]
                    tempImage[x][y] = tempnewPixelVal         # to keep the same format as array values

    plt.imshow(tempImage,cmap='gray')
    plt.show()          # this image will have the filtered component with seperate brightness


    ## to find centroid of the object

    def findCentroid(labelValue, equivalenceDict, keyEquidict, tempImage):
        Xi = 0          # total of X co-ordinates values
        Yj = 0          # total of Y co-ordinated values
        startPoint = []
        for x in range(0,512,1):
            for y in range(0,512,1):
                if startPoint == [] : #and tempImage[x][y] == labelValue :
                    if tempImage[x][y] == labelValue:
                        startPoint = [x,y]
                        Xi = x
                        Yj = y
                if tempImage[x][y] == labelValue:
                    Xi = Xi + x
                    Yj = Yj + y

        area = equivalenceDict.get(keyEquidict)
        x = Xi / (area[0])        # x - co-ordinate of centroid
        y = Yj / (area[0])        # y - co-ordinate of centroid

        print('     Below are the co-ordinates of the centroid for the respective component label:')
        print('     Xc :', math.floor(x), 'Yc:', math.floor(y))
        return math.floor(x),math.floor(y)

    ## to find the bounding box

    def findBoundingBox(labelValue, tempImage):
        startPoint = []
        endPoint = []
        Xmin = 0
        Ymin = 0
        Xmax = 0
        Ymax = 0
        for x in range(0,512,1):
            for y in range(0,512,1):
                if startPoint == []:
                    if tempImage[x][y] == labelValue:
                        startPoint = [x,y]
                        Xmin = x
                        Ymin = y
                        Xmax = x
                        Ymax = y

                if tempImage[x][y] == labelValue:
                    endPoint = [x,y]
                    if y < Ymin:
                       Ymin = y
                    if x > Xmax:
                        Xmax = x
                    if x < Xmin:
                        Xmin = x
                    if y > Ymax:
                        Ymax = y

        #Xmin = startPoint[0]     # x co-ordinate of starting point
        #Ymax = endPoint[1]       # y co-ordinate of end point

        print('     (Xmin, Ymin):',Xmin,',',Ymin)
        print('     (Xmax, Ymax):',Xmax,',',Ymax)


    ## to find the boundary of the object

    def detectBoundary(labelValue, tempImage):
        #boundaryDataImage = tempImage

        startPoint = []
        endPoint = []
        Xmin = 0
        Ymin = 0
        Xmax = 0
        Ymax = 0
        # to determine the bounding box to minimize the computation

        for x in range(0,512,1):
            for y in range(0,512,1):
                if startPoint == []:
                    if tempImage[x][y] == labelValue:
                        startPoint = [x,y]
                        Xmin = x
                        Ymin = y
                        Xmax = x
                        Ymax = y

                if tempImage[x][y] == labelValue:
                    endPoint = [x,y]
                    if y < Ymin:
                       Ymin = y
                    if x > Xmax:
                        Xmax = x
                    if x < Xmin:
                        Xmin = x
                    if y > Ymax:
                        Ymax = y

        current = startPoint
        b = []
        x = current[0]
        y = current[1]
        b = [x, y - 1]

        perimeter = 0
        flag = 0
        index = 1

        while True :

            eight_neighbourhoodlist = [[x, y - 1],[x-1,y-1],[x-1,y],[x-1,y+1],[x,y+1],[x+1,y+1],[x+1,y],[x+1,y-1]]

            index = index - 1
            if index < 0:
                index = 7

            while True:

                if eight_neighbourhoodlist[index][0] == 512 or eight_neighbourhoodlist[index][1] == 512 or\
                        eight_neighbourhoodlist[index][0] < 0 or eight_neighbourhoodlist[index][1] > 511 or \
                        eight_neighbourhoodlist[index][1] < 0:

                    pass
                else:
                    if tempImage[eight_neighbourhoodlist[index][0]][eight_neighbourhoodlist[index][1]] == labelValue:
                    # assign the pixel as current pixel
                        current = eight_neighbourhoodlist[index]
                        x = current[0]
                        y = current[1]
                        perimeter = perimeter + 1
                        b = eight_neighbourhoodlist[index - 1]
                        break

                index = index + 1

                if index > 7:
                    index = 0

            if startPoint == current:
                break

        return perimeter


    # Calculate axis of elongation and eccentricity
    def axisOfElomgation(labelValue, equivalenceDict, keyEquidict, tempImage):
        Xc = 0  # x co-ordinate of centroid
        Yc = 0  # y co-ordinate of centroid
        a = 0
        b = 0
        c = 0

        Xc , Yc = findCentroid(labelValue,equivalenceDict,keyEquidict,tempImage)
        #print('inside the axis para : ',Xc,Yc)
        # calculate a, b, c
        for x in range(0,512,1):
            for y in range(0,512,1):
                if tempImage[x][y] == labelValue :
                    xDASHij = x - Xc
                    yDASHij = y - Yc

                    a = a + xDASHij
                    c = c + yDASHij
                    b = b + xDASHij * yDASHij

        # calculate final b

        b = 2 * b   # since the summation is 2 times

        # calculate angle in degrees

        theota = math.degrees(math.atan(b / ( a - c )))
        theota = theota / 2

        sin2theota = math.sin(2 * theota)
        cos2theota = math.cos(2 * theota)

        Xintertia1 = 1/2 * ( a + c ) + 1/2 * ( a - c ) * -1 * cos2theota + 1/2 * b * -1 * sin2theota
        Xintertia2 = 1/2 * ( a + c ) + 1/2 * ( a - c ) * cos2theota + 1/2 * b * sin2theota

        if Xintertia1 < Xintertia2:
            XintertiaMIN = Xintertia1
            XintertiaMAX = Xintertia2
        else:
            XintertiaMIN = Xintertia2
            XintertiaMAX = Xintertia1

        if XintertiaMIN < 0:
            #change sign for computation
            XintertiaMIN = -1 * XintertiaMIN

        if XintertiaMAX < 0:
            #chnage sign for computation
            XintertiaMAX = -1 * XintertiaMAX

        print('     The second order moments:',' a = ',a,', b = ',b,' c = ',c)
        print('     X\u00b2min:', XintertiaMIN)
        print('     X\u00b2max:', XintertiaMAX)

        # orientation of axis of elongation

        if sin2theota < 0:
            valsin2theota = -1 * sin2theota
        else:
            valsin2theota = sin2theota

        if cos2theota < 0:
            valcos2theota = -1 * cos2theota
        else:
            valcos2theota = cos2theota

        print('     orientation of axis of elongation :')
        print('     sin 2\u0398 = \u00B1',valsin2theota)
        print('     cos 2\u0398 = \u00B1',valcos2theota)

        eccentricity = XintertiaMAX / XintertiaMIN

        return eccentricity



    #### 2.b Description of each component in terms of ####
    #       1. The component size i.e area
    #       2. The location of the centroid
    #       3. The coordinates of the bounding box
    #       4. The orientation of the axis of elongation
    #       5. The eccentricity, perimeter and compactness

    componentCount = 1

    for filteredComponent in graphicDisplayListImageC:
        print('Description for component ', componentCount, ' in terms of :')

        area = equivalenceDict.get(filteredComponent)
        print('1. Area = ', area[0])

        label = graphicDisplayDictImageC.get(filteredComponent)
        # centroid
        print('2. The location of the centroid :')
        x, y = findCentroid(label[0], equivalenceDict, filteredComponent, tempImage)

        # co-ordinates of bouding box
        print('3. The co-ordinated of the bouding box:')
        findCentroid(label[0], equivalenceDict, filteredComponent, tempImage)

        # the orientation of axis of elongation
        print('4. The orientation of the axis of elongation:')
        eccentricity = axisOfElomgation(label[0], equivalenceDict, filteredComponent, tempImage)

        # the eccentricity, perimeter and compactness
        print('5. The eccentricity, perimeter and compactness:')
        print('     eccentricity = ', eccentricity)
        perimeter = detectBoundary(label[0], tempImage)
        print('     perimeter = ', perimeter)
        compactness = (perimeter * perimeter) / area[0]
        print('     compactness = ', compactness)

        componentCount = componentCount + 1

# call to the CCL Algorithm

IterativeCCL(BTimage,7000)


####    Iterative Conncected Component Labelling Algorithm Definition

def IterativeCCL(BTimage, sizeFilterValue):
    label = 5   # initialization of the label varaible
    i = 0
    j = 1
    equivalenceList = [[5]]

    tempImage = np.array(BTimage,dtype=int)

    #if value <= 128 then tempImage == 2
    for x in range(513):
        for y in range(513):
            if tempImage[x][y] == 2:
                if tempImage[x - 1][y] != tempImage[x][y - 1] and min(tempImage[x][y - 1], tempImage[x - 1][y]) > 0 and min(x,y) >= 1:
                    tempimage[x][y] = min(tempImage[x - 1][y], tempImage[x][y - 1])
                        #logic using for loop to append the label
                        for item in equivalenceList:
                            for subItem in item:
                                if subItem == tempImage[x-1][y] and tempImage[x][y-1]!= 2 and tempImage[x][y-1] not in item:
                                        item.append(tempImage[x][y - 1])
                                        break
                                pass  # pass for inner loop if
                            pass  # pass for outer loop if
                else:
                    if tempImage[x - 1][y] > 0 and x >= 1:
                        tempImage[x][y] = tempImage[x - 1][y]
                    else:
                        if tempImage[x][y - 1] > 0 and y >= 1:
                            tempImage[x][y] = tempImage[x][y - 1]
                        else:
                            tempImage[x][y] = label
                            label = label + 1
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

    for x in range(513):
        for y in range(513):
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

    for x in range(512):
        for y in range(512):
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

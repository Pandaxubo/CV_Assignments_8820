
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

### starts the algorithm
def IterCCL(image, FilterValue):

    ### set width and height 
    w, h = 512, 512

    ### throw the first 512 pixels
    with open(image, 'rb') as f: 
        f.seek(512) 
        img = np.fromfile(f, dtype=np.uint8).reshape((h,w)) 

    ### plot the oringinal image(B) 
    plt.imshow(img)
    plt.show()

    ### threshold image followed by instruction
    ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) 

    ### make a zero matrix to store each pixel's label(named matrix)
    height,width = img.shape[0],img.shape[1]
    matrix = np.zeros((height,width))

    ### plot the threshold image(Bt) 
    plt.imshow(thresh1)
    plt.show()

    ###algorithm implementation
    label = 1 ### label start from 1
    mergeDic = {} ### contains pairs that need to be merged
    area = {} ### contains filtered component and their area

    for x in range(height):
        for y in range(width):
            value = thresh1[x,y]
            if value == 0:
                left = matrix[x,y-1]
                top  = matrix[x-1,y] ### used to judge the current pixel
                if left > 0 and top > 0:
                    matrix[x,y] = min(left,top)
                    if left != top:
                        mergeDic.update([ ( max(left,top),min(left,top) ) ])
                elif left > 0:
                    matrix[x,y] = left
                elif top > 0:
                    matrix[x,y] = top
                else: 
                    label = label+1
                    matrix[x,y] = label
                
    key = list(mergeDic.keys())
    val = list(mergeDic.values())

    for i in range(0,label+1): ### initalize area
        area[i] = 0   

    for x in range(0,height): ### start merging label
        for y in range(0,width):
            values = matrix[x,y]
            for z in range(len(key)):
                if values == key[z]:
                    matrix[x,y] = val[z]
            area[matrix[x,y]] = area[matrix[x,y]] + 1 
                
        

    DisplayArea = []       ### contains filtered labels
    for key, value in area.items():
        if value >= FilterValue and value <= 100000:
            DisplayArea.append(key)


    print('Total Number of filtered Components: ' , len(DisplayArea))
    print('Filterted components labels: ', DisplayArea)

    ### find centroid of the object(equation is: x = xsum/ area, y = ysum/area)
    def findCentroid(Component, area, matrix):
        Xsum = 0         
        Ysum = 0          
        areaNum = area.get(Component)
        PointForBegin = []

        for x in range(height):
            for y in range(width):
                if PointForBegin == [] : 
                    if matrix[x,y] == Component:
                        PointForBegin = [x,y]
                        Xsum = x
                        Ysum = y
                if matrix[x,y] == Component:
                    Xsum = Xsum + x
                    Ysum = Ysum + y

        finalX = Xsum / (areaNum)        
        finalY = Ysum / (areaNum)       
        return math.floor(finalX),math.floor(finalY)

    ### find the bounding box of each component(the greatest and smallest x and y value in the target component)
    def findBoundingBox(Component, matrix):
        PositionForUp = 0
        PositionForDown = 0
        PositionForLeft = 0
        PositionForRight = 0
        flag = 0
        for x in range(height):
            for y in range(width):
                if matrix[x, y] == Component:
                    if flag == 0:
                        PositionForUp = y
                        PositionForDown = y
                        PositionForLeft = x
                        PositionForRight = x
                        flag = flag + 1
                    else: 
                        PositionForUp = max(PositionForUp , y)
                        PositionForDown = min(PositionForDown , y)
                        PositionForLeft = max(PositionForLeft , x)
                        PositionForRight = min(PositionForRight , x)
        return PositionForUp, PositionForDown, PositionForLeft , PositionForRight

    ### plot the final image(C) 
    plt.imshow(matrix)
    plt.show()


    ### show component details
    componentCount = 1

    for Component in DisplayArea:
        print('Component', componentCount, 'details:')

        areaNum = area.get(Component)
        print('1. The component size:', areaNum)
        
        x, y = findCentroid(Component, area, matrix)
        print('2. The location of the centroid:', '(',x,',' ,y,')')

        print('3. The coordinates of the bounding box:')
        y1, y2, x1, x2 = findBoundingBox(Component, matrix)
        print('The bounding box is:','(',x1,',',y1,')',';','(',x1,',',y2,')',';','(',x2,',',y1,')',';','(',x2,',',y2,')',';')

        componentCount = componentCount + 1 

### run the algorithm, the value means the range of components(min area)
IterCCL("Data/comb.img", 6000)
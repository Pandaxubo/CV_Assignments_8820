import numpy as np
from matplotlib import pyplot as plt
import math
from collections import OrderedDict
import cv2 as cv

class Assignment2:

    def __init__(self, image_path):
        self.image_path = image_path

    def _get_binary_image(self):
        ### set width and height 
        ### throw the first 512 pixels
        with open(self.image_path, 'rb') as f: 
            f.seek(512) 
            img = np.fromfile(f, dtype=np.uint8).reshape((512,512)) 

        ### plot the oringinal image B 
        plt.imshow(img)
        plt.show()

        ### threshold image followed by instruction
        # Threshold the image for the value T = 128 to generate the binary image Bt
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] <= 128 ) :
                    img[i][j] = 255
                else:
                    img[i][j] = 0 

        ### plot the threshold image(Bt) 
        plt.imshow(img,'gray')
        plt.show()
        self.Bt = img


    def _modify_image_to_zero_one(self):
        #set all values of pixels to 1 which are not 0
        for x in range(0,512,1):
            for y in range(0,512,1):
                if self.Bt[x][y] != 0:
                    self.Bt[x][y] = 1

    def _is_not_boundary(self, x, y):
        if x > 0 and y > 0 and x < 511 and y < 511:
            return True
        else:
            return False

    def _get_four_neighbors(self, x, y):
        return [int(self.Bt[x][y]), int(self.Bt[x - 1][y]), int(self.Bt[x][y + 1]),
                                                int(self.Bt[x + 1][y]), int(self.Bt[x][y - 1])]


    def _calculate_distance_transform(self):
        for x in range(512):
            for y in range(512):
                if self.Bt[x][y] != 0 and self._is_not_boundary(x, y):
                    self.Bt[x][y] = 1 + min(self._get_four_neighbors(x, y))


        # generate skeleton
    def _get_skeleton(self):
        self._modify_image_to_zero_one()

        BTimageold = self.Bt
        tempImage1 = np.copy(BTimageold)
        self._calculate_distance_transform()
        while not np.array_equal(tempImage1,self.Bt):
            BTimageold = self.Bt
            tempImage1 = np.copy(BTimageold)
            self._calculate_distance_transform()
        print("Bt is: ")
        plt.imshow(self.Bt,cmap='gray')
        plt.show()

        skeleton = np.zeros((512, 512), dtype= np.uint8)

        for x in range(512):
            for y in range(512):
                if self.Bt[x][y] != 0 and self._is_not_boundary(x, y):
                    localmaxima = max(self._get_four_neighbors(x, y))
                    if int(self.Bt[x][y]) >= localmaxima:
                        skeleton[x][y] = self.Bt[x][y]
        print("M is: ")
        plt.imshow(skeleton,cmap='gray')
        plt.show()

        self.skeleton = skeleton
        return 

    def _update_four_neighbors(self, x, y, newValue):
        self.tempImg[x - 1][y] = newValue if newValue >= self.tempImg[x - 1][y] else self.tempImg[x - 1][y]
        self.tempImg[x + 1][y] = newValue if newValue >= self.tempImg[x + 1][y] else self.tempImg[x + 1][y]
        self.tempImg[x][y - 1] = newValue if newValue >= self.tempImg[x][y - 1] else self.tempImg[x][y - 1]
        self.tempImg[x][y + 1] = newValue if newValue >= self.tempImg[x][y + 1] else self.tempImg[x][y + 1]
                                                
    def _rebuild_image(self):
        tempImg = np.copy(self.skeleton)
        self.tempImg = tempImg

        maxElement = 0
        for x in range(0,512,1):
            for y in range(0,512,1):
                if tempImg[x][y] > maxElement :
                    maxElement = tempImg[x][y]
        
        
        for n in range(maxElement):
            for i in range(512):
                for j in range(512):
                    if self.tempImg[i][j] != 0 and self._is_not_boundary(i, j):
                        tempValue = self.tempImg[i][j] - 1
                        if tempValue >= 0: 
                            self._update_four_neighbors(i, j, tempValue)

                            
        for x in range(0,512,1):
            for y in range(0,512,1):
                if tempImg[x][y] != 0 :
                    tempImg[x][y] = 255
        plt.imshow(tempImg, cmap='gray')
        plt.show()

result = Assignment2("Data/comb.img")
result._get_binary_image()
result._get_skeleton()
result._rebuild_image()
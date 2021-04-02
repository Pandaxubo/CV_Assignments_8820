import numpy as np
from matplotlib import pyplot as plt
import math
import cv2 as cv
from numpy.lib.histograms import histogramdd

class Assignment4:

    def __init__(self,image_path):
        self.image_path = image_path    ### init image path 
        self.w = 512                    ### width
        self.h = 512                    ### height

    ### read original image
    def _get_original_image(self):
        with open(self.image_path, 'rb') as f: 
            f.seek(512) ### throw the first 512 pixels
            img = np.fromfile(f, dtype=np.uint8).reshape((512,512)) ###reshape image

        ### plot the original image B 
        print("B is:")
        plt.imshow(img, cmap = 'gray')
        plt.show()

        return img

    ### display image (If Threshold is False, that means image is thresholded.)
    def _display_image(self, String, T, img, threshold : bool):
        img = self._threshold_image(T, img)
        print(String)
        plt.imshow(img, cmap='gray')
        plt.show()

    
### run class here
result = Assignment4("Data/test1.img")  ### replace path to input other images
img = result._get_original_image()  ### get oringinal image B



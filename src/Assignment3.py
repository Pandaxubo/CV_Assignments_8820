import numpy as np
from matplotlib import pyplot as plt
import math
import cv2 as cv
from numpy.lib.histograms import histogramdd

class Assignment3:

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

        self.B = img

    ### threshold image
    def _threshold_image(self, threshold):
        target = np.copy(self.B)
        for x in range(self.w):
            for y in range(self.h):
                if target[x][y] >= threshold:
                    target[x][y] = 255
                else:
                    target[x][y] = 0
        return target

    ########################################################################
    ### 1. Thresholding using peakiness detection
    def _peakiness_detection(self):

        ### generate histogram
        histogram = np.round(np.transpose(cv.calcHist([self.B], [0], None, [256], [0,256]))).astype(int)[0].tolist()

        ### store final peakiness
        peakiness_threshold = 0

        ### evaluation criterion to select peaks(easily to calculate peakiness)
        peaks = []
        for item in histogram:
            if item >= 2000: 
                peaks.append(item)

        ### find gi and gj(Duplicate tuples exist, like (a, b) and (b, a), but it will not affect result)
        gi_gj = {}     ### store gi and gj
        for i in range(1, len(peaks)-1, 1):
            gi_gj[histogram.index(peaks[i])]= histogram.index(self._select_neighbor(peaks, i))
        gi_gj[histogram.index(peaks[len(peaks) - 1])] = histogram.index(peaks[len(peaks) - 2])

        ### find gk for each(gi, gj)
        gk = []
        for key in gi_gj:
            gk.append(self._find_lowest_point(key, gi_gj[key], histogram))

        ### calculate peakiness list
        peakiness = []
        i = 0
        for key in gi_gj:
            peakiness.append(min(histogram[key], histogram[gi_gj[key]]) / gk[i])
            i += 1

        ### get highest peakiness from list above
        T = max(peakiness)

        ### display image
        peakinessed_image = self._threshold_image(T)
        print("After peakiness:")
        plt.imshow(peakinessed_image, cmap='gray')
        plt.show()

    ### select neighbor for each peak for constructing gi_gj(For peakiness thresholding)
    def _select_neighbor(self, peaks, i):
        return peaks[i - 1] if abs(peaks[i - 1] - peaks[i]) > abs(peaks[i + 1] - peaks[i]) else peaks[i + 1]

    ### find lowest point between gi and gj(For peakiness thresholding)
    def _find_lowest_point(self, gi, gj, histogram):
        point = histogram[min(gi, gj)]
        for k in range(min(gi, gj), max(gi, gj)):
            point = min(histogram[k], point)
        return point

    ########################################################################
    ### 2. Iterative thresholding
    def _iterative_thresholding(self):

        ### initial estimate of T is the average of intensity values in B 
        estimate_T = math.ceil(np.mean(self.B))
        
        ### initialize a copy of B  
        temp_B = np.copy(self.B)

        ### calculate threshold. iter_list[0, 1, 2] respond to u1, u2 and new_T for each iteration
        iter_list = self._iter_T(temp_B, estimate_T)
        former_u1 = 0
        former_u2 = 0
        while former_u1 != iter_list[0] and former_u2 != iter_list[1]:  
            former_u1 = iter_list[0]
            former_u2 = iter_list[1]
            iter_list = self._iter_T(temp_B, iter_list[2])
        
        ### get T in final iteration
        T = round(iter_list[2])

        ### display image
        thresholded_image = self._threshold_image(T)
        print("After iterative thresholding:")
        plt.imshow(thresholded_image, cmap='gray')
        plt.show()
        
    ### do iterations to find final T (For iterative thresholding)
    def _iter_T(self, temp_B, current_T):

        ### initialize R1, R2 to store split area
        R1 = []
        R2 = []
        for x in range(self.w):
            for y in range(self.h):
                if temp_B[x][y] < current_T:
                    R1.append(temp_B[x][y])
                else:
                    R2.append(temp_B[x][y])

        ### get u1, u2 responding to R1 and R2
        u1 = math.ceil(np.mean(R1))
        u2 = math.ceil(np.mean(R2))
        new_T = (u1 + u2) / 2

        return [u1, u2, new_T]


result = Assignment3("Data/test1.img")
result._get_original_image()
result._peakiness_detection()
result._iterative_thresholding()


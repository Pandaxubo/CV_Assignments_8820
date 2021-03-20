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

        return img

    ### threshold image
    def _threshold_image(self, threshold, img):
        target = np.copy(img)
        for x in range(target.shape[0]):
            for y in range(target.shape[1]):
                if target[x][y] >= threshold:
                    target[x][y] = 255
                else:
                    target[x][y] = 0

        return target

    ### display image (If Threshold is False, that means image is thresholded.)
    def _display_image(self, String, T, img, threshold : bool):
        if threshold == True:
            img = self._threshold_image(T, img)
        print(String)
        plt.imshow(img, cmap='gray')
        plt.show()

    ########################################################################
    ### 1. Thresholding using peakiness detection
    def _peakiness_detection(self, img):

        ### generate histogram
        histogram = np.round(np.transpose(cv.calcHist([img], [0], None, [256], [0,256]))).astype(int)[0].tolist()

        ### store final peakiness
        T = 0

        ### evaluation criterion to select peaks(easily to calculate peakiness)
        peaks = []
        for item in histogram:
            if item >= 2000: 
                peaks.append(item)

        ### find gi and gj(Duplicate tuples exist, like (a, b) and (b, a), but it will not affect result)
        gi_gj = {}     ### store gi and gj
        for i in range(1, len(peaks)-1, 1):
            gi_gj[histogram.index(peaks[i])]= histogram.index(self._select_neighbor(peaks, histogram, i))
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
        self._display_image("Peakiness Detection", T, img, True)

    ### select neighbor for each peak for constructing gi_gj(For peakiness thresholding)
    def _select_neighbor(self, peaks, histogram, i):

        return peaks[i + 1] if abs(histogram.index(peaks[i - 1]) - histogram.index(peaks[i])) > abs(histogram.index(peaks[i + 1]) - histogram.index(peaks[i])) else peaks[i - 1]

    ### find lowest point between gi and gj(For peakiness thresholding)
    def _find_lowest_point(self, gi, gj, histogram):
        point = histogram[min(gi, gj)]
        for k in range(min(gi, gj), max(gi, gj)):
            point = min(histogram[k], point)

        return point

    ########################################################################
    ### 2. Iterative thresholding(Display is explained in the code)
    def _iterative_thresholding(self, img, display: bool):

        ### initial estimate of T is the average of intensity values in B 
        estimate_T = math.ceil(np.mean(img))
        
        ### initialize a copy of B  
        temp_B = np.copy(img)

        ### calculate threshold. iter_list[0, 1, 2] respond to u1, u2 and new_T for each iteration
        iter_list = self._iter_T(temp_B, estimate_T)
        former_u1 = 0
        former_u2 = 0
        while former_u1 != iter_list[0] and former_u2 != iter_list[1]:  
            former_u1 = iter_list[0]
            former_u2 = iter_list[1]
            iter_list = self._iter_T(temp_B, iter_list[2])
        
        ### get T in final iteration
        T = math.ceil(iter_list[2])

        ### display image
        if display == True:
            self._display_image("Iterative Thresholding", T, img, True)

        ### This part is for Adaptive Thresholding. Display function called in that part. 
        if display == False:
            return self._threshold_image(T, img)
        
    ### do iterations to find final T (For iterative thresholding)
    def _iter_T(self, temp_B, current_T):

        ### initialize R1, R2 to store split area
        R1 = []
        R2 = []
        for x in range(temp_B.shape[0]):
            for y in range(temp_B.shape[1]):
                if temp_B[x][y] < current_T:
                    R1.append(temp_B[x][y])
                else:
                    R2.append(temp_B[x][y])

        ### get u1, u2 responding to R1 and R2
        u1 = math.ceil(np.mean(R1))
        u2 = math.ceil(np.mean(R2))
        new_T = (u1 + u2) / 2

        return [u1, u2, new_T]

    ########################################################################
    ### 3. Adaptive thresholding(split original image to 4 equal size sub-images)
    def _adaptive_thresholding(self, img):
        
        ### split 4 sub-images and merge to target image
        sub1, sub2, sub3, sub4 = self._split_image(img)
        sub1 = self._iterative_thresholding(sub1, False)
        sub2 = self._iterative_thresholding(sub2, False)
        sub3 = self._iterative_thresholding(sub3, False)
        sub4 = self._iterative_thresholding(sub4, False)
        target_image = self._merge_image(sub1, sub2, sub3, sub4)

        ### display image. Thresholding value is useless since it has been thresholded when merging it.
        self._display_image("Adaptive thresholding", 0, target_image, False)


    ### split original image to 4 equal size sub-images(For adaptive thresholding)
    def _split_image(self, img):
        
        ### if we don`t use int here will raise error
        self.half_w = int(self.w / 2)
        self.half_h = int(self.h / 2)
        self.full_w = int(self.w)
        self.full_h = int(self.h)

        ### initialize 4 sub-images
        sub1 = np.zeros((self.half_w, self.half_h), dtype='uint8')
        sub2 = np.zeros((self.half_w, self.half_h), dtype='uint8')
        sub3 = np.zeros((self.half_w, self.half_h), dtype='uint8')
        sub4 = np.zeros((self.half_w, self.half_h), dtype='uint8')

        ### value 4 sub-images
        for x in range(self.half_w):
            for y in range(self.half_h):
                sub1[x][y] = img[x][y]

        for x in range(self.half_w, self.full_w):
            for y in range(0, self.half_h):
                sub2[x - self.half_w][y] = img[x][y]

        for x in range(self.half_w):
            for y in range(self.half_h, self.full_h):
                sub3[x][y - self.half_h] = img[x][y]

        for x in range(self.half_w, self.full_w):
            for y in range(self.half_h, self.full_h):
                sub4[x - self.half_w][y - self.half_h] = img[x][y]

        return sub1, sub2, sub3, sub4

    ### merge 4 sub-images into target image(For adaptive thresholding)
    def _merge_image(self, sub1, sub2, sub3, sub4):

        ### initialize target threshold image
        target_image = np.zeros((self.full_w, self.full_h), dtype='uint8')

        ### value target image
        for x in range(self.half_w):
            for y in range(self.half_h):
                target_image[x][y] = sub1[x][y]

        for x in range(self.half_w, self.full_w):
            for y in range(self.half_h):
                target_image[x][y] = sub2[x - self.half_w][y]

        for x in range(self.half_w):
            for y in range(self.half_h, self.full_h):
                target_image[x][y] = sub3[x][y - self.half_h]

        for x in range(self.half_w, self.full_w):
            for y in range(self.half_h, self.full_h):
                target_image[x][y] = sub4[x - self.half_w][y - self.half_h]
        
        return target_image

    ########################################################################
    ### 4. Dual thresholding with region growing
    def _dual_thresholding(self, img):

        ### initial estimate of T1 and T2. Here I choose the average and 75% intensity as T1 and T2.
        T1 = math.ceil(np.mean(img))
        T2 = np.percentile(img, 75)

        ### initialize R1, R2, R3 to store split area
        R1 = np.zeros((self.full_w, self.full_h), dtype='uint8')
        R2 = np.zeros((self.full_w, self.full_h), dtype='uint8')
        R3 = np.zeros((self.full_w, self.full_h), dtype='uint8')

        ### value 3 regions with T1 and T2
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y] < T1:
                    R1[x][y] = img[x][y]
                elif img[x][y] >= T1 and img[x][y] <= T2:
                    R2[x][y] = img[x][y]
                elif img[x][y] > T2:
                    R3[x][y] = img[x][y]

        ### if a pixel in R2 has a neighbor in R1, then assgin that pixel to R1
        for i in range(R2.shape[0]):
            for j in range(R2.shape[1]):
                if self._is_not_boundary(i, j) and R2[i][j] != 0:
                    if R1[i - 1][j] != 0 and R1[i][j - 1] != 0 and R1[i + 1][j] != 0 and R1[i][j + 1] != 0:
                        R1[i][j] = R2[i][j]
                        R2[i][j] = 0

        ### after detection above, remove all pixels in R2 to R3 
        for x in range(R2.shape[0]):
            for y in range(R2.shape[1]):
                if R2[x][y] != 0 :
                    R3[x][y] = R2[x][y]
                    R2[x][y] = 0

        ### Convert core region into binary image. According to textbook, we only need to threshold this part.
        for x in range(R1.shape[0]):
            for y in range(R1.shape[1]):
                if R1[x][y] != 0:
                    R1[x][y] = 255

        ### display image. Thresholding value is useless since it has been thresholded when merging it.
        self._display_image("Dual Thresholding", 0, R1, False)
                        
    ### abandon boundary pixels(For dual thresholding)
    def _is_not_boundary(self, x, y):
        if x > 0 and y > 0 and x < 511 and y < 511:
            return True
        else:
            return False

### run class here
result = Assignment3("Data/test1.img")  ### replace path to input other images
img = result._get_original_image()  ### get oringinal image B
result._peakiness_detection(img)    ### Part 1
result._iterative_thresholding(img, True)   ### Part 2
result._adaptive_thresholding(img)  ### Part 3
result._dual_thresholding(img)  ### Part 4


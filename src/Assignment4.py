import numpy as np
from matplotlib import pyplot as plt
import math

class Assignment4:

    def __init__(self,image_path):
        self.image_path = image_path    ### init image path 
        self.w = 512                    ### width
        self.h = 512                    ### height
        self.sigma = 5                  ### sigma

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

    ### display image 
    def _display_image(self, img, padding):

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] != 0 :
                    img[i][j] = 255

        tar = np.zeros(shape=(512,512))

        for i in range(tar.shape[0]):
            for j in range(tar.shape[1]):
                tar[i][j] = img[i + padding][j + padding]
        plt.imshow(tar, cmap='gray')
        plt.show()

    def _construct_filter(self):
        ### define the size of the filter(4 * sigma)
        size = int(2 * (np.ceil(4 * self.sigma)) + 1)

        ### define the values in the filter(-4 * sigma, 4 * sigma)
        # x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1), np.arange(-size / 2 + 1, size / 2 + 1))
        x, y = np.meshgrid(np.ceil(np.arange(-size / 2, size / 2)), np.ceil(np.arange(-size / 2, size / 2)))

        # Construct LoG filter(2D)
        filter = (((x ** 2 + y ** 2 - (2.0 * self.sigma ** 2)) / self.sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2.0 * self.sigma ** 2))) / (1 / (2.0 * np.pi * self.sigma ** 2))

        filter_size= filter.shape[0]

        return filter, filter_size

    def _add_padding(self):
        
        filter, filter_size = self._construct_filter()

        ### add padding to the input image
        padding = math.floor(filter_size/2)

        temp_image = np.zeros((img.shape[0] + 2 * padding, img.shape[1] + 2 * padding), dtype=np.uint8)
        
        ### fill non-padding area to new image
        for i in range(temp_image.shape[0]):
            for j in range(temp_image.shape[1]):
                if i >= padding and i < self.w + padding and j >= padding and j < self.h + padding:
                    temp_image[i][j] = img[i - padding][j - padding]

        return temp_image, filter_size, filter, padding

    def _run_filter(self):

        temp_image, filter_size, filter, padding = self._add_padding()

        temp_zeros = np.zeros(temp_image.shape, dtype=float) 

        aaron = temp_image.shape[0]-(filter_size - 1)
        # apply filter
        for i in range(temp_image.shape[0]-(filter_size - 1)):          
            for j in range(temp_image.shape[1]-(filter_size - 1)):
                # if i + filter_size  < self.w and j + filter_size  < self.h:
                #if i < self.w and j < self.h:
                    window = temp_image[i: i + filter_size, j:j + filter_size] * filter
                    temp_zeros[i + padding, j + padding] = np.sum(window)

        temp_zeros = temp_zeros.astype(np.int64, copy=False)

        zero_crossing = np.zeros(temp_zeros.shape)

        #find zero crossings
        for i in range(temp_zeros.shape[0]):
            for j in range(temp_zeros.shape[1]):
                if temp_zeros[i][j] == 0:
                    # to check for vertical or horizontal
                    if (temp_zeros[i][j - 1] < 0 and temp_zeros[i][j + 1] > 0) or (temp_zeros[i][j - 1] < 0 and temp_zeros[i][j + 1] < 0) or (
                            temp_zeros[i - 1][j] < 0 and temp_zeros[i + 1][j] > 0) or (temp_zeros[i - 1][j] > 0 and temp_zeros[i + 1][j] < 0):
                        zero_crossing[i][j] = 255
                if temp_zeros[i][j] < 0:
                    if (temp_zeros[i][j - 1] > 0) or (temp_zeros[i][j + 1] > 0) or (temp_zeros[i - 1][j] > 0) or (temp_zeros[i + 1][j] > 0):
                        zero_crossing[i][j] = 255

        return zero_crossing, padding, temp_image, filter_size, filter,temp_zeros

    def _get_zero_crossing(self):
        zero_crossing, padding, temp_image, filter_size, filter, temp_zeros = self._run_filter()
        # for i in range(zero_crossing.shape[0]):
        #     for j in range(zero_crossing.shape[1]):
        #         if zero_crossing[i][j] == 255 and (i + filter_size) < zero_crossing.shape[0] and (j + filter_size) \
        #                 < zero_crossing.shape[1] and (i + padding) < zero_crossing.shape[0] and (j + padding) < zero_crossing.shape[1]:
        #             # convolve around the same pixel in the original image and its 8-neighbors
        #             window = temp_image[i: i + filter_size, j:j + filter_size] * filter
        #             temp_zeros[i + padding, j + padding] = np.sum(window)
        #             # 8-neighbors

        #             # i-1 ; j-1
        #             window = temp_image[i-1: i-1 + filter_size, j-1:j-1 + filter_size] * filter
        #             temp_zeros[i-1 + padding, j-1 + padding] = np.sum(window)

        #             # i-1 ; j
        #             window = temp_image[i - 1: i - 1 + filter_size, j:j + filter_size] * filter
        #             temp_zeros[i - 1 + padding, j + padding] = np.sum(window)

        #             # i-1 ; j+1
        #             window = temp_image[i - 1: i - 1 + filter_size, j + 1:j + 1 + filter_size] * filter
        #             temp_zeros[i - 1 + padding, j + 1 + padding] = np.sum(window)

        #             # i ; j-1
        #             window = temp_image[i : i + filter_size, j - 1:j - 1 + filter_size] * filter
        #             temp_zeros[i + padding, j - 1 + padding] = np.sum(window)

        #             # i-1 ; j+1
        #             window = temp_image[i - 1: i - 1 + filter_size, j + 1:j + 1 + filter_size] * filter
        #             temp_zeros[i - 1 + padding, j + 1 + padding] = np.sum(window)

        #             # i+1 ; j-1
        #             window = temp_image[i + 1: i + 1 + filter_size, j - 1:j - 1 + filter_size] * filter
        #             temp_zeros[i + 1 + padding, j - 1 + padding] = np.sum(window)

        #             # i+1 ; j
        #             window = temp_image[i + 1: i + 1 + filter_size, j:j + filter_size] * filter
        #             temp_zeros[i + 1 + padding, j + padding] = np.sum(window)

        #             # i+1 ; j+1
        #             window = temp_image[i + 1: i + 1 + filter_size, j + 1:j + 1 + filter_size] * filter
        #             temp_zeros[i + 1 + padding, j + 1 + padding] = np.sum(window)

        # temp_zeros = temp_zeros.astype(np.int64, copy=False)

        # target = np.zeros_like(temp_zeros)

        # #find zero crossings
        # for i in range(temp_zeros.shape[0]):# - (filter_size - 1)):
        #     for j in range(temp_zeros.shape[1]):#- (filter_size - 1)):
        #         if temp_zeros[i][j] == 0:
        #             # to check for vertical or horizontal
        #             if (temp_zeros[i][j - 1] < 0 and temp_zeros[i][j + 1] > 0) or (temp_zeros[i][j - 1] < 0 and temp_zeros[i][j + 1] < 0) or (
        #                 temp_zeros[i - 1][j] < 0 and temp_zeros[i + 1][j] > 0) or (temp_zeros[i - 1][j] > 0 and temp_zeros[i + 1][j] < 0):
        #                 target[i][j] = 255
        #         if temp_zeros[i][j] < 0:
        #             if (temp_zeros[i][j - 1] > 0) or (temp_zeros[i][j + 1] > 0) or (temp_zeros[i - 1][j] > 0) or (temp_zeros[i + 1][j] > 0):
        #                 target[i][j] = 255

        return target, padding

    
### run class here
result = Assignment4("Data/test3.img")  ### replace path to input other images
img = result._get_original_image()  ### get oringinal image B
result._run_filter()
target, padding = result._get_zero_crossing()
result._display_image(target, padding)



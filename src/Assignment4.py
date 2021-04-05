import numpy as np
from matplotlib import pyplot as plt
import math

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

    ### display image 
    def _display_image(self, img, f):

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] != 0 :
                    img[i][j] = 255
        
        plt.title('sigma = %f' %f)
        plt.imshow(img, cmap='gray')
        plt.show()

    def _construct_filter(self, sigma):

        ### define the size of the filter(4 * sigma)
        size = int(2 * (np.ceil(4 * sigma)) + 1)

        ### define the values in the filter(-4 * sigma, 4 * sigma)
        x, y = np.meshgrid(np.ceil(np.arange(-size / 2, size / 2)), np.ceil(np.arange(-size / 2, size / 2)))

        ### construct LoG filter(2D)
        filter = (((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (1 / (2.0 * np.pi * sigma ** 2))
        filter_size= filter.shape[0]

        ### add padding to the input image
        padding = math.floor(filter_size/2)

        ### initialize image to be filtered (padding added)
        temp_image = np.zeros((img.shape[0] + 2 * padding, img.shape[1] + 2 * padding), dtype=np.uint8)
        
        ### fill non-padding area to new image
        for i in range(temp_image.shape[0]):
            for j in range(temp_image.shape[1]):
                if i >= padding and i < self.w + padding and j >= padding and j < self.h + padding:
                    temp_image[i][j] = img[i - padding][j - padding]

        return temp_image, filter_size, filter, padding

    def _run_filter(self, sigma):

        temp_image, filter_size, filter, padding = self._construct_filter(sigma)

        temp_zeros = np.zeros(temp_image.shape, dtype=float) 

        ### do convolution
        for i in range(temp_image.shape[0]-(filter_size - 1)):          
            for j in range(temp_image.shape[1]-(filter_size - 1)):
                    window = temp_image[i: i + filter_size, j:j + filter_size] * filter
                    temp_zeros[i + padding, j + padding] = np.sum(window)

        temp_zeros = temp_zeros.astype(np.int64, copy=False)

        ### store zero crossing
        zero_crossing = np.zeros(temp_zeros.shape)

        ### contains three situations: vertical, horizontal, and diagonal
        for i in range(1,temp_zeros.shape[0]-1):
            for j in range(1,temp_zeros.shape[1]-1):
                if temp_zeros[i][j] <= 0:
                    if (temp_zeros[i][j-1] * temp_zeros[i][j+1] <0) or (temp_zeros[i-1][j] * temp_zeros[i+1][j] <0) or \
                    (temp_zeros[i-1][j+1] * temp_zeros[i+1][j-1] <0) or (temp_zeros[i+1][j+1] * temp_zeros[i-1][j-1] <0):
                        zero_crossing[i][j] = 255
        
        ### reshape target image 
        result = np.zeros(shape=(512,512))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = zero_crossing[i+padding][j+padding]

        return result, padding

    
### run class here
result = Assignment4("Data/test3.img")  ### replace path to input other images
img = result._get_original_image()  ### get oringinal image B

target = np.zeros(shape=(512,512))

for i in np.arange(5, 0, -0.5): ### only print 5 images as requirement
    target_temp, padding = result._run_filter(i)
    target = target + target_temp
    if(i%1==0):
        result._display_image(target,i)




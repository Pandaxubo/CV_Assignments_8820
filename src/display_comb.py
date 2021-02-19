import cv2 
import numpy as np 
import matplotlib.pyplot as pl
from PIL import Image

#image1 = cv2.imread('Data/comb.img') 
shape = (512,512)
dtype = np.dtype(np.uint8)
f = open('Data/comb.img')

f.seek(512)
data = np.fromfile(f, dtype)

i = data.reshape(shape)
pl.imshow(i)
pl.show()

cv2.imwrite("output.jpg", i)

image = cv2.imread("output.jpg")
cv2.imshow('1',image)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
#image.show()
# applying different thresholding 
# techniques on the input image 
# all pixels value above 120 will 
# be set to 255 
thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) 


# the window showing output images 
# with the corresponding thresholding 
# techniques applied to the input images 
cv2.imshow('Binary Threshold', thresh1) 
	
# De-allocate any associated memory usage 
if cv2.waitKey(0) & 0xff == 27: 
	cv2.destroyAllWindows() 

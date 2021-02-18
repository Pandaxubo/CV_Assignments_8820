import cv2 
import numpy as np 
import matplotlib.pyplot as pl

#image1 = cv2.imread('Data/comb.img') 
shape = (512,512)
dtype = np.dtype(np.int8)
f = open('Data/comb.img')
f.seek(512)
data = np.fromfile(f, dtype)

i = data.reshape(shape)
cv2.imwrite("output.jpg", i)
image = cv2.imread("output1.jpg") 
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
image.show()
# applying different thresholding 
# techniques on the input image 
# all pixels value above 120 will 
# be set to 255 
ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) 
#ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 
#ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC) 
#ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO) 
#ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV) 

# the window showing output images 
# with the corresponding thresholding 
# techniques applied to the input images 
cv2.imshow('Binary Threshold', thresh1) 
#cv2.imshow('Binary Threshold Inverted', thresh2) 
#cv2.imshow('Truncated Threshold', thresh3) 
#cv2.imshow('Set to 0', thresh4) 
#cv2.imshow('Set to 0 Inverted', thresh5) 
	
# De-allocate any associated memory usage 
if cv2.waitKey(0) & 0xff == 27: 
	cv2.destroyAllWindows() 

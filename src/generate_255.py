import numpy as np
from PIL import Image
import cv2

filename = 'test_data/comb.img' 

# set width and height 
w, h = 512, 512

with open(filename, 'rb') as f: 
    # Seek backwards from end of file by 2 bytes per pixel 
    f.seek(512) 
    img = np.fromfile(f, dtype=np.uint8).reshape((h,w)) 

# Save as PNG, and retain 16-bit resolution
#Image.fromarray(img).save('test_data/temp.png')

#image = cv2.imread('test_data/temp.png')
# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# applying different thresholding 
# techniques on the input image 
# all pixels value above 128 will 
# be set to 255 
ret, thresh1 = cv2.threshold(img, 128, 0, cv2.THRESH_BINARY) 
#cv2.imshow('Binary Threshold', thresh1) 
cv2.imwrite('test_data/thresd1.jpg', thresh1)

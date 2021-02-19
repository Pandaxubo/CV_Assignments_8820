import numpy as np
from PIL import Image
import cv2

filename = 'Data/comb.img' 

# set width and height 
w, h = 512, 512

with open(filename, 'rb') as f: 
    # Seek backwards from end of file by 2 bytes per pixel 
    f.seek(512) 
    img = np.fromfile(f, dtype=np.uint8).reshape((h,w)) 

# Save as PNG, and retain 16-bit resolution
Image.fromarray(img).save('Data/temp.png')

# Alternative to line above - save as JPEG, but lose 16-bit resolution
# Image.fromarray((img>>8).astype(np.uint8)).save('result.jpg') 
image = cv2.imread("Data/temp.png")
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# applying different thresholding 
# techniques on the input image 
# all pixels value above 120 will 
# be set to 255 
ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) 
#cv2.imshow('Binary Threshold', thresh1) 
cv2.imwrite('Data/thresd.png', thresh1)

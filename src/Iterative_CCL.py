import os
import numpy as np
import cv2

source = os.path.join('Data', 'thresd.png')
destPath = os.path.join('Data','comb_final.png')

img = cv2.imread(source,0)
img = np.array(img)

threshold_val = 128
img = cv2.threshold(img, threshold_val, 1, cv2.THRESH_BINARY)[1]

def getNeighbors(img, i, j):
	"""
		Computes the 4 visted neighbors of a pixel(i,j)
		Returns a list of neighbors
	"""
	H, W = img.shape
	neighbors = []
	if i-1 >= 0:
		neighbors.append(img[i-1][j])
	if j-1 >= 0:
		neighbors.append(img[i][j-1])
	if j-1 >= 0 and i-1 >= 0:
		neighbors.append(img[i-1][j-1])
	if j+1 < W and i-1 >= 0:
		neighbors.append(img[i-1][j+1])
	return neighbors

def connectedComponents(image):
	"""
		Receives an image for which a labelled image is computed
		O(n**2) Complexity
		Returns a tuple of (labelled_img, equivalence_dict)
	"""
	equiv = {}
	H, W = image.shape
	labels = np.zeros_like(image)
	label = 1
	
	for i in range(0, H):
		for j in range(0, W):

			if image[i][j] == 1:
				neighbors = getNeighbors(labels, i, j)
				# print(neighbors)
				neighbors = list(filter(lambda a: a != 0, neighbors))
				

				if len(neighbors) == 0:
					labels[i][j] = label
					equiv[label] = set([label])
					label += 1
					
				else:
					minVal = min(neighbors)
					labels[i][j] = minVal
					for l in neighbors:
						equiv[l] = set.union(equiv[l], neighbors)
	finalLabels = {}
	newLabel = 1
	for i in range(H):
		for j in range(W):
			if labels[i][j] != 0:
				new = find(labels[i][j], equiv)
				labels[i][j] = new

				if new not in finalLabels:
					finalLabels[new] = newLabel
					newLabel += 1

	for i in range(H):
		for j in range(W):
			if labels[i][j] != 0:
				labels[i][j] = finalLabels[labels[i][j]]

	return labels, equiv

def find(label, equiv):
	"""
		Finds the root label in equivalence classes of labels
		Parameters: 
			label -> int
			Value of a labelled pixel, to traverse for finding the root label
			equiv -> dict
			Contains equivalence classes for each label
		Return:
			minVal -> int
			The root label
	"""
	minVal = min(equiv[label])
	while label != minVal:
		label = minVal
		minVal = min(equiv[label])
	
	return minVal

def imshow_components(labels, show=False):
	# Map component labels to hue val
	label_hue = np.uint8(100*labels) #/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
	# set bg label to black
	labeled_img[label_hue==0] = 0
	if show:
		cv2.imshow('labeled.png', labeled_img)
		cv2.waitKey()
	return labeled_img
	
	
	

if __name__ == "__main__":
	labelled, equiv = connectedComponents(img)
	labeled_img = imshow_components(labelled)
	cv2.imwrite(destPath, labeled_img)
	print("---DONE---")


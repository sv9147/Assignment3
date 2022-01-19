import cv2, sys
from skimage import feature
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.preprocess import Preprocess
from skimage import color
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import prewitt_h,prewitt_v,sobel

class LBP:
	def __init__(self, num_points=8, radius=2, eps=1e-6, resize=100):
		self.num_points = num_points * radius
		self.radius = radius
		self.eps = eps
		self.resize=resize


	def extract(self, img):
		
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		
		lbp = feature.local_binary_pattern(img, self.num_points, self.radius, method="nri_uniform")
		
		
		
		n_bins = int(lbp.max() + 1)
		hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
		
		
		return hist		
		

	def edge_feature_extract(self, img): 
		
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		
		edge_canny = feature.canny(img, sigma=1)
		edge_harris = corner_peaks(corner_harris(img), min_distance=1)
		edge_h = prewitt_h(img)
		edge_v = prewitt_v(img) 
		
		edge_sobel = sobel(img)
		
		n_bins = int(edge_v.max() + 1)
		hist, _ = np.histogram(edge_v, density=True, bins=n_bins, range=(0, n_bins))
		
		
		return hist

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Extractor()
	features = extractor.extract(img)
	print(features)
